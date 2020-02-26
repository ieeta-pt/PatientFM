import time
import torch
import pickle
import numpy as np
from torch import nn
from sklearn import metrics

from models.BiLstmCRF.utils import generate_batch, classListToTensor, update_progress, value_to_key
from models.BiLstmCRF.decoder import Decoder
from models.BiLstmCRF.encoder import BiLSTMEncoder


class Model:
    def __init__(self, configs, labels_dict, max_length, device):
        self.device = device
        self.max_length = max_length
        self.entity_classes = labels_dict
        self.output_size = len(self.entity_classes)
        self.hidden_size = configs.hidden_size
        self.decoder_insize = self.hidden_size * 2
        self.batch_size = configs.batch_size
        self.num_layers = configs.num_layers
        self.embedding_dim = configs.WORDVEC_SIZE
        self.epochs = configs.epochs
        self.iterations_per_epoch = configs.iterations_per_epoch
        self.learning_rate = configs.learning_rate

        self.encoder = BiLSTMEncoder(self.embedding_dim, self.hidden_size, self.output_size, batch_size=self.batch_size, num_layers=self.num_layers).to(device=self.device)
        self.decoder = Decoder(self.decoder_insize, self.hidden_size, self.output_size, batch_size=self.batch_size, max_len=self.max_length, num_layers=self.num_layers).to(device=self.device)
        self.criterion = nn.NLLLoss()
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


    def train(self, train_tokenized_sentences, train_sentences_embeddings, train_labels):

        r""" List with entity labels ("O":0 maps to nonentity) """
        entity_labels = [value for key, value in self.entity_classes.items() if not key.startswith('O')]

        n_iterations = self.epochs * self.iterations_per_epoch
        current_loss = 0
        all_losses = []
        f1_score_accumulated = 0

        start = time.clock()
        global_start = start
        for iteration in range(n_iterations):
            sentences_tensor, sorted_batch_classes, sorted_len_units, packed_input, mask = \
                generate_batch(train_tokenized_sentences, train_sentences_embeddings, self.embedding_dim, train_labels, self.batch_size, self.device)

            initial_state_h0c0 = self.encoder.initH0C0(self.device)
            encoder_output, hidden_state_n, cell_state_n = self.encoder(packed_input.to(device=self.device), initial_state_h0c0)
            crf_out, entity_out, crf_loss = self.decoder(encoder_output, sorted_batch_classes, mask, self.device)

            entity_pred, entity_true = self.compute_entity_prediction(entity_out, sorted_len_units, sorted_batch_classes)
            current_loss += self.perform_optimization_step(crf_loss, entity_pred, entity_true)
            label_pred, label_true = self.compute_label_prediction(crf_out, sorted_len_units, sorted_batch_classes)

            f1_score_accumulated += metrics.f1_score(label_pred.tolist(), label_true.tolist(), average='micro', labels=entity_labels)
            if (iteration + 1) % self.iterations_per_epoch == 0:
                all_losses.append(current_loss / self.iterations_per_epoch)
                print("Epoch %d of %d | F1: %.3f%% | Average Loss: %.5f" %
                      ((iteration + 1) / self.iterations_per_epoch, self.epochs, f1_score_accumulated / self.iterations_per_epoch * 100, current_loss / self.iterations_per_epoch))
                current_loss = 0
                f1_score_accumulated = 0
                elapsed = (time.clock() - start)
                print("Time used:", elapsed)
                start = time.clock()

        elapsed = (time.clock() - global_start)
        print("Completed training. Total time used: {}".format(elapsed))
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()


    def train_time_debug(self, train_tokenized_sentences, train_sentences_embeddings, train_labels):

        r""" List with entity labels ("O":0 maps to nonentity) """
        entity_labels = [value for key, value in self.entity_classes.items() if not key.startswith('O')]

        n_iterations = self.epochs * self.iterations_per_epoch
        current_loss = 0
        all_losses = []
        f1_score_accumulated = 0

        start = time.clock()
        global_start = start

        debug_prev_time = time.clock()

        for iteration in range(n_iterations):
            sentences_tensor, sorted_batch_classes, sorted_len_units, packed_input, mask = \
                generate_batch(train_tokenized_sentences, train_sentences_embeddings, self.embedding_dim, train_labels, self.batch_size, self.device)

            debug_time = time.clock()
            print("Time to generate batch: {}".format(debug_time - debug_prev_time))
            debug_prev_time = debug_time

            initial_state_h0c0 = self.encoder.initH0C0(self.device)

            debug_time = time.clock()
            print("Time to initialize encoder: {}".format(debug_time - debug_prev_time))
            debug_prev_time = debug_time

            encoder_output, hidden_state_n, cell_state_n = self.encoder(packed_input.to(device=self.device), initial_state_h0c0)

            debug_time = time.clock()
            print("Time for encoder step: {}".format(debug_time - debug_prev_time))
            debug_prev_time = debug_time

            crf_out, entity_out, crf_loss = self.decoder(encoder_output, sorted_batch_classes, mask, self.device)

            debug_time = time.clock()
            print("Time for decoder step: {}".format(debug_time - debug_prev_time))
            debug_prev_time = debug_time

            entity_pred, entity_true = self.compute_entity_prediction(entity_out, sorted_len_units, sorted_batch_classes)

            debug_time = time.clock()
            print("Time for Entity prediction: {}".format(debug_time - debug_prev_time))
            debug_prev_time = debug_time

            current_loss += self.perform_optimization_step(crf_loss, entity_pred, entity_true)

            debug_time = time.clock()
            print("Time for optimization step: {}".format(debug_time - debug_prev_time))
            debug_prev_time = debug_time

            label_pred, label_true = self.compute_label_prediction(crf_out, sorted_len_units, sorted_batch_classes)

            debug_time = time.clock()
            print("Time for Label prediction: {} \n\n".format(debug_time - debug_prev_time))


            f1_score_accumulated += metrics.f1_score(label_pred.tolist(), label_true.tolist(), average='micro', labels=entity_labels)
            if (iteration + 1) % self.iterations_per_epoch == 0:
                all_losses.append(current_loss / self.iterations_per_epoch)
                print("Epoch %d of %d | F1: %.3f%% | Average Loss: %.5f" %
                      ((iteration + 1) / self.iterations_per_epoch, self.epochs, f1_score_accumulated / self.iterations_per_epoch * 100, current_loss / self.iterations_per_epoch))
                current_loss = 0
                f1_score_accumulated = 0
                elapsed = (time.clock() - start)
                print("Time used:", elapsed)
                start = time.clock()

        elapsed = (time.clock() - global_start)
        print("Completed training. Total time used: {}".format(elapsed))
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()


    def test(self, test_tokenized_sentences, test_sentences_embeddings, test_labels):
        test_label_true = []
        test_label_pred = []

        for sentence_idx, tokenized_sentence in enumerate(test_tokenized_sentences):
            x_input = torch.zeros((1, len(tokenized_sentence), self.embedding_dim)).to(self.device)
            mask = torch.zeros((1, len(tokenized_sentence)), dtype=torch.long).to(self.device)
            mask[0] = 1

            r""" Convert numpy array with embeddings to tensor, and reshape to (sentence length x Embedding size)"""
            sentence_length = len(tokenized_sentence)
            sentence_tensor = torch.from_numpy(test_sentences_embeddings[sentence_idx, :sentence_length * self.embedding_dim]).float()
            sentence_tensor = sentence_tensor.view(sentence_length, self.embedding_dim)

            x_input[0] = sentence_tensor
            x_input = nn.utils.rnn.pack_padded_sequence(x_input, torch.tensor([x_input.shape[1]]), batch_first=True)
            y_true_tensor = [test_labels[sentence_idx].to(self.device)]
            initial_state_h0c0 = (torch.zeros((2*self.num_layers, 1, self.hidden_size), dtype=torch.float32, device=self.device),
                                  torch.zeros((2*self.num_layers, 1, self.hidden_size), dtype=torch.float32, device=self.device))

            encoder_output, hidden_state_n, cell_state_n = self.encoder(x_input.to(device=self.device), initial_state_h0c0)
            crf_out, entity_out, crf_loss = self.decoder(encoder_output, y_true_tensor, mask, self.device)
            test_label_pred.extend(crf_out[0])
            test_label_true.extend(y_true_tensor[0][0].tolist())

            if (sentence_idx + 1) % 100 == 0:
                update_progress(sentence_idx / len(test_tokenized_sentences))
        update_progress(1)

        return test_label_pred, test_label_true


    def compute_entity_prediction(self, entity_out, sorted_len_units, sorted_batch_classes):
        label_num = 0
        entity_pred = torch.zeros((sorted_len_units.sum(), 2), dtype=torch.float, device=self.device)
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


    def compute_label_prediction(self, crf_out, sorted_len_units, sorted_batch_classes):
        label_num = 0
        label_pred = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=self.device)
        label_true = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=self.device)

        for batch_sample_idx in range(self.batch_size):
            sentence_len = sorted_len_units[batch_sample_idx]
            crf_out_sample = crf_out[batch_sample_idx]
            label_sample_pred = crf_out_sample[:sentence_len]
            for position in range(sentence_len.item()):
                label_true[label_num] = sorted_batch_classes[batch_sample_idx][0][position]
                label_pred[label_num] = label_sample_pred[position]
                label_num += 1

        return label_pred, label_true


    def perform_optimization_step(self, crf_loss, entity_pred, entity_true):
        entity_loss = self.criterion(entity_pred,entity_true)
        loss = crf_loss + 0.2 * entity_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item()


    def get_entity_type_count(self, testLabels):
        r""" This process works by trimming the first part of the label string (B-/I-/P-/M-/NA-),
        resulting in the label with the entity type
        POSITIVE_CLASSES: B-Observation, I-Observation, P-FamilyMember, M-FamilyMember, NA-FamilyMember"""

        entityDict = {}
        for labelIdx, testLabel in enumerate(testLabels):
            label = value_to_key(testLabel, self.entity_classes)
            labelPrefix = label[:2]
            if labelPrefix in ("P-", "M-", "NA"):
                entityType = "FamilyMember"
                entityDict[(labelIdx, labelIdx)] = entityType
            elif labelPrefix == "B-":
                entityType = "Observation"
                nextLabelIdx = labelIdx + 1
                nextLabel = value_to_key(testLabels[nextLabelIdx], self.entity_classes)
                if nextLabel != "I-Observation":
                    entityDict[(labelIdx, labelIdx)] = entityType
                elif nextLabel == "I-Observation":
                    while nextLabel == "I-Observation":
                        nextLabelIdx += 1
                        if nextLabelIdx <= (len(testLabels)-1):
                            nextLabel = value_to_key(testLabels[nextLabelIdx], self.entity_classes)
                        else:
                            nextLabel = "EndSentence"
                    entityDict[(labelIdx, nextLabelIdx-1)] = entityType

        entityTypeCount = {'Observation': 0, 'FamilyMember': 0}
        for _, entityType in entityDict.items():
            entityTypeCount[entityType] += 1

        return entityDict, entityTypeCount


    def evaluate_test(self, testLabelPred, testLabelTrue):
        labels = [value for key, value in self.entity_classes.items()]# if key != "O"] # NO MODELO ORIGINAL USAVAM COM ESTA CONDIÇÃO, SEM A CLASSE 0
        print(metrics.classification_report(testLabelTrue, testLabelPred, labels=labels))

        trueEntity, trueEntityTypeCount = self.get_entity_type_count(testLabelTrue)
        predEntity, predEntityTypeCount = self.get_entity_type_count(testLabelPred)

        exactHits = {'Observation': 0, 'FamilyMember': 0}
        for key, entity_type in trueEntity.items():
            if key in predEntity.keys() and predEntity[key] == entity_type:
                exactHits[entity_type] += 1

        entityTypes = ['Observation', 'FamilyMember']
        recall = [exactHits[entityType] / trueEntityTypeCount[entityType] for entityType in entityTypes]
        precision = [exactHits[entityType] / predEntityTypeCount[entityType] for entityType in entityTypes]
        fp = np.mean([(2 * precision[i] * recall[i]) / (precision[i] + recall[i]) for i in range(len(entityTypes))])
        print("Exact Hit metrics: F1 = {} | Precision = {} | Recall = {}".format(round(fp, 4), round(np.mean(precision), 4), round(np.mean(recall), 4)))


    def write_model_files(self, test_label_pred, test_label_true, seed):
        timepoint = time.strftime("%Y%m%d_%H%M")
        path = '../results/models/BiLstmCRF/'
        filename = 'N2C2_BioWordVec_' + timepoint + '-' + str(seed)
        pickle.dump([test_label_pred, test_label_true], open(path + "results-" + filename + '.pickle', 'wb'))
        torch.save(self.encoder, path + 'model_encoder-'+filename)
        torch.save(self.decoder, path + 'model_decoder-'+filename)

def loadModelConfigs(settings):
    class Args:
        pass
    configs = Args()
    configs.epochs = int(settings["DLmodelparams"]["epochs"])
    configs.iterations_per_epoch = int(settings["DLmodelparams"]["iterationsperepoch"])
    configs.hidden_size = int(settings["DLmodelparams"]["hiddensize"])
    configs.batch_size = int(settings["DLmodelparams"]["batchsize"])
    configs.num_layers = int(settings["DLmodelparams"]["numlayers"])
    configs.learning_rate = float(settings["DLmodelparams"]["learningrate"])
    configs.WORDVEC_SIZE = int(settings["embeddings"]["wordvec_size"])
    return configs
