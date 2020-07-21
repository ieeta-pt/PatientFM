import os
import time
import torch
import pickle
import numpy as np
from torch import nn
from sklearn import metrics

from transformers import AutoTokenizer, AutoModel

from models.utils import classListToTensor, updateProgress, valueToKey
from models.Embedding_BiLstmCRF.utils import generateBatch, concatenateNejiClassesToEmbeddings
from models.BiLstmCRF.decoder import Decoder
from models.BiLstmCRF.encoder import BiLSTMEncoder

class Model:
    def __init__(self, configs, labels_dict, max_length, device):
        self.device = device
        self.ENTITY_PREDICTION = configs.ENTITY_PREDICTION
        self.max_length = max_length
        self.entity_classes = labels_dict
        self.output_size = len(self.entity_classes)
        self.hidden_size = configs.hidden_size
        self.decoder_insize = self.hidden_size * 2
        self.batch_size = configs.batch_size
        self.num_layers = configs.num_layers
        self.epochs = configs.epochs
        self.iterations_per_epoch = configs.iterations_per_epoch
        self.learning_rate = configs.learning_rate

        self.clinicalBERT = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device=self.device)
        self.bert_outsize = 768

        self.USE_NEJI = configs.USE_NEJI
        if self.USE_NEJI == "True":
            self.bert_outsize = self.bert_outsize + 1
        # else:
        #     self.encoder_insize = self.bert_outsize

        self.linear = nn.Linear(self.bert_outsize, self.output_size).to(device=self.device)
        self.softmax = nn.LogSoftmax(dim=2).to(device=self.device)


        self.entity_criterion = nn.NLLLoss()
        self.label_criterion = nn.NLLLoss()
        self.linear_optimizer = torch.optim.Adam(self.linear.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


        # self.encoder = BiLSTMEncoder(self.encoder_insize, self.hidden_size, self.output_size, batch_size=self.batch_size, num_layers=self.num_layers).to(device=self.device)
        # self.decoder = Decoder(self.decoder_insize, self.hidden_size, self.output_size, batch_size=self.batch_size, max_len=self.max_length, num_layers=self.num_layers).to(device=self.device)
        # self.criterion = nn.NLLLoss()
        # self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    def train(self, train_encoded_sentences_tensors, train_labels, neji_classes=None):

        r""" List with entity labels ("O":0 maps to nonentity) """
        entity_labels = [value for key, value in self.entity_classes.items() if not key.startswith('O')]

        n_iterations = self.epochs * self.iterations_per_epoch
        current_loss = 0
        all_losses = []
        f1_score_accumulated = 0

        start = time.time()
        global_start = start

        TRAINING = True
        loss_delta = 0.5 # corresponds to 0.5% change in value
        current_patience = 0
        while TRAINING:
            for iteration in range(n_iterations):

                if self.USE_NEJI == "True":
                    encoded_sentences_tensors, sorted_batch_classes, sorted_len_units, mask, batch_neji_classes = \
                        generateBatch(train_encoded_sentences_tensors, train_labels, self.batch_size, self.device, neji_classes)
                else:
                    encoded_sentences_tensors, sorted_batch_classes, sorted_len_units, mask = \
                        generateBatch(train_encoded_sentences_tensors, train_labels, self.batch_size, self.device)

                sentence_representations = self.clinicalBERT(input_ids=encoded_sentences_tensors, attention_mask=mask)
                # Output from clinicalBERT model is a tuple with (last_hidden_state, pooler_output), so we select the first part
                sentence_representations = sentence_representations[0]

                if self.USE_NEJI == "True":
                    sentence_representations = concatenateNejiClassesToEmbeddings(sentence_representations, batch_neji_classes, self.device)
                #     packed_input = torch.nn.utils.rnn.pack_padded_sequence(sentence_neji_embeddings, sorted_len_units, batch_first=True)
                # else:
                #     packed_input = torch.nn.utils.rnn.pack_padded_sequence(sentence_representations, sorted_len_units, batch_first=True)

                entity_out = self.linear(sentence_representations)
                entity_out = self.softmax(entity_out)

                if self.ENTITY_PREDICTION == "True":
                    entity_pred, entity_true = self.compute_entity_prediction(entity_out, sorted_len_units, sorted_batch_classes)
                    label_pred, label_true, prediction_probs = self.compute_label_prediction(entity_out, sorted_len_units, sorted_batch_classes)
                    current_loss += self.perform_optimization_step_entity_pred(entity_pred, entity_true, prediction_probs, label_true)
                elif self.ENTITY_PREDICTION == "False":
                    label_pred, label_true, prediction_probs = self.compute_label_prediction(entity_out, sorted_len_units, sorted_batch_classes)
                    current_loss += self.perform_optimization_step(prediction_probs, label_true)

                f1_score_accumulated += metrics.f1_score(label_pred.tolist(), label_true.tolist(), average='micro', labels=entity_labels)
                if (iteration + 1) % self.iterations_per_epoch == 0:
                    all_losses.append(current_loss / self.iterations_per_epoch)
                    print("Epoch %d of %d | F1: %.3f%% | Average Loss: %.5f" %
                          ((iteration + 1) / self.iterations_per_epoch, self.epochs, f1_score_accumulated / self.iterations_per_epoch * 100, current_loss / self.iterations_per_epoch))
                    current_loss = 0
                    f1_score_accumulated = 0
                    elapsed = (time.time() - start)
                    print("Time used in seconds:", elapsed)
                    start = time.time()

                    # If training is completed
                    if (iteration + 1) == n_iterations:
                        TRAINING=False
                        break

                    if len(all_losses) > 1:
                        loss_change = (all_losses[-1]-all_losses[-2])/all_losses[-2]*100
                        if abs(loss_change) < loss_delta:
                            current_patience += 1
                            print("Last loss change was smaller than {}%. Patience increased to {}.".format(loss_delta, current_patience))
                            if current_patience == self.patience:
                                print("Reached patience of {}. Stopping training.".format(self.patience))
                                TRAINING=False
                                break
                        else:
                            current_patience = 0

        elapsed = (time.time() - global_start)
        print("Completed training. Total time used: {}".format(elapsed))



    def test(self, test_encoded_sentences_tensors, test_labels, verbose=False, neji_classes=None, SINGLE_INSTANCE=False):
        test_label_true = []
        test_label_pred = []

        for sentence_idx, encoded_sentence_tensor in enumerate(test_encoded_sentences_tensors):
            sentence_length = encoded_sentence_tensor.shape[0]
            x_input = torch.zeros((1, sentence_length), dtype=torch.long).to(self.device)
            mask = torch.zeros((1, sentence_length), dtype=torch.long).to(self.device)
            mask[0] = 1

            r""" Convert numpy array with embeddings to tensor, and reshape to (sentence length x Embedding size)"""
            if SINGLE_INSTANCE is True:
                y_true_tensor = [test_labels.to(self.device)]
            else:
                y_true_tensor = [test_labels[sentence_idx].to(self.device)]

            x_input[0] = encoded_sentence_tensor
            sentence_representation = self.clinicalBERT(input_ids=x_input, attention_mask=mask)
            sentence_representation = sentence_representation[0]

            if neji_classes is not None:
                if SINGLE_INSTANCE is True:
                    neji_tensor = neji_classes.to(self.device)
                else:
                    neji_tensor = neji_classes[sentence_idx].to(self.device)

                sentence_representation = concatenateNejiClassesToEmbeddings(sentence_representation, neji_tensor, self.device)

            entity_out = self.linear(sentence_representation)
            entity_out = self.softmax(entity_out)
            entity_out = entity_out.max(2)[1] #Similar to argmax, but backpropagable

            test_label_pred.extend(entity_out[0].tolist())
            test_label_true.extend(y_true_tensor[0][0].tolist())

            if ((sentence_idx + 1) % 100 == 0) and verbose:
                updateProgress(sentence_idx / len(test_encoded_sentences_tensors))
        if verbose:
            updateProgress(1)

        return test_label_pred, test_label_true

    def compute_entity_prediction(self, entity_out, sorted_len_units, sorted_batch_classes):
        label_num = 0
        entity_pred = torch.zeros((sorted_len_units.sum(), len(self.entity_classes)), dtype=torch.float, device=self.device)
        entity_true = torch.zeros((sorted_len_units.sum()), dtype=torch.long, device=self.device)

        for batch_sample_idx in range(self.batch_size):
            sentence_len = sorted_len_units[batch_sample_idx]
            entity_out_sample = entity_out[batch_sample_idx]
            entity_sample_pred = entity_out_sample[:sentence_len, :]
            for position in range(sentence_len.item()):
                if sorted_batch_classes[batch_sample_idx][0][position] != self.entity_classes['O']:
                    entity_true[label_num] = 1
                else:
                    entity_true[label_num] = 0
                entity_pred[label_num] = entity_sample_pred[position, :]
                label_num += 1

        return entity_pred, entity_true

    def compute_label_prediction(self, linear_out, sorted_len_units, sorted_batch_classes):
        label_num = 0
        label_pred = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=self.device)
        label_true = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=self.device)
        predictionProbs = torch.zeros((sorted_len_units.sum(), len(self.entity_classes)),dtype=torch.float,device=self.device)

        classPredictions = linear_out.max(2)[1]

        for batch_sample_idx in range(self.batch_size):
            sentence_len = sorted_len_units[batch_sample_idx]
            classPredictions_sample = classPredictions[batch_sample_idx]
            label_sample_pred = classPredictions_sample[:sentence_len]
            predictionProbs_sample = linear_out[batch_sample_idx]
            predictionProbs_pred = predictionProbs_sample[:sentence_len]
            for position in range(sentence_len.item()):
                label_true[label_num] = sorted_batch_classes[batch_sample_idx][0][position]
                label_pred[label_num] = label_sample_pred[position]
                predictionProbs[label_num] = predictionProbs_pred[position]
                label_num += 1

        return label_pred, label_true, predictionProbs

    def perform_optimization_step_entity_pred(self, entity_pred, entity_true, prediction_probs, label_true):
        entity_loss = self.entity_criterion(entity_pred,entity_true)
        label_loss  = self.label_criterion(prediction_probs, label_true)
        loss = 0.8 * label_loss + 0.2 * entity_loss
        self.linear_optimizer.zero_grad()
        loss.backward()
        self.linear_optimizer.step()
        return loss.item()

    def perform_optimization_step(self, prediction_probs, label_true):
        loss = self.label_criterion(prediction_probs, label_true)
        self.linear_optimizer.zero_grad()
        loss.backward()
        self.linear_optimizer.step()
        return loss.item()


    def get_entity_type_count(self, testLabels):
        r""" This process works by trimming the first part of the label string (B-/I-/BP/BM/BN),
        resulting in the label with the entity type
        POSITIVE_CLASSES: B-Observation, I-Observation, BP-FamilyMember, BM-FamilyMember, BNA-FamilyMember,
        IP-FamilyMember, IM-FamilyMember, INA-FamilyMember"""

        entityDict = {}
        for labelIdx, testLabel in enumerate(testLabels):
            label = valueToKey(testLabel, self.entity_classes)
            labelPrefix = label[:2]
            if labelPrefix == "B-":
                entityType = "Observation"
                nextLabelIdx = labelIdx + 1
                nextLabel = valueToKey(testLabels[nextLabelIdx], self.entity_classes)
                if nextLabel != "I-Observation":
                    entityDict[(labelIdx, labelIdx)] = entityType
                elif nextLabel == "I-Observation":
                    while nextLabel == "I-Observation":
                        nextLabelIdx += 1
                        if nextLabelIdx <= (len(testLabels)-1):
                            nextLabel = valueToKey(testLabels[nextLabelIdx], self.entity_classes)
                        else:
                            nextLabel = "EndSentence"
                    entityDict[(labelIdx, nextLabelIdx-1)] = entityType
            elif labelPrefix == "BP" or labelPrefix == "BM" or labelPrefix == "BN":
                entityType = "FamilyMember"
                nextLabelIdx = labelIdx + 1
                nextLabel = valueToKey(testLabels[nextLabelIdx], self.entity_classes)
                if nextLabel != "IP-FamilyMember" and nextLabel != "IM-FamilyMember" and nextLabel != "INA-FamilyMember":
                    entityDict[(labelIdx, labelIdx)] = entityType
                elif nextLabel == "IP-FamilyMember" or nextLabel == "IM-FamilyMember" or nextLabel == "INA-FamilyMember":
                    while nextLabel == "IP-FamilyMember" or nextLabel == "IM-FamilyMember" or nextLabel == "INA-FamilyMember":
                        nextLabelIdx += 1
                        if nextLabelIdx <= (len(testLabels)-1):
                            nextLabel = valueToKey(testLabels[nextLabelIdx], self.entity_classes)
                        else:
                            nextLabel = "EndSentence"
                    entityDict[(labelIdx, nextLabelIdx-1)] = entityType

        entityTypeCount = {'Observation': 0, 'FamilyMember': 0}
        for _, entityType in entityDict.items():
            entityTypeCount[entityType] += 1

        return entityDict, entityTypeCount


    def evaluate_test(self, testLabelPred, testLabelTrue):
        labels = [value for key, value in self.entity_classes.items() if key != "O"] # NO MODELO ORIGINAL USAVAM COM ESTA CONDIÇÃO, SEM A CLASSE 0
        print(metrics.classification_report(testLabelTrue, testLabelPred, labels=labels))

        trueEntity, trueEntityTypeCount = self.get_entity_type_count(testLabelTrue)
        predEntity, predEntityTypeCount = self.get_entity_type_count(testLabelPred)

        exactHits = {'Observation': 0, 'FamilyMember': 0}
        for key, entity_type in trueEntity.items():
            if key in predEntity.keys() and predEntity[key] == entity_type:
                exactHits[entity_type] += 1

        entityTypes = ['Observation', 'FamilyMember']
        recall = []
        precision = []
        for entityType in entityTypes:
            try:
                tempRecall = exactHits[entityType] / trueEntityTypeCount[entityType]
            except ZeroDivisionError:
                print("Found a ZeroDivisionError in Recall. Recall value was set to a residual value.")
                tempRecall = 0.00001
            finally:
                recall.append(tempRecall)

            try:
                tempPrecision = exactHits[entityType] / predEntityTypeCount[entityType]
            except ZeroDivisionError:
                print("Found a ZeroDivisionError in Precision. Precision value was set to a residual value.")
                tempPrecision = 0.00001
            finally:
                precision.append(tempPrecision)

        fp = np.mean([(2 * precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(len(entityTypes))])
        print("Exact Hit metrics: F1 = {} | Precision = {} | Recall = {}".format(round(fp, 4), round(np.mean(precision), 4), round(np.mean(recall), 4)))


    def write_model_files(self, seed):
        timepoint = time.strftime("%Y%m%d_%H%M")
        path = '../results/models/clinicalBERT/'
        if not os.path.exists(path):
            os.makedirs(path)
        filename = 'N2C2_clinicalBERT_' + timepoint + '-seed_' + str(seed)
        torch.save(self.linear.state_dict(), path + 'model_linear-'+filename+".pth")


    def load_model_files(self, linear_path):
        self.linear.load_state_dict(torch.load(linear_path, map_location=self.device))
        self.linear.eval()
