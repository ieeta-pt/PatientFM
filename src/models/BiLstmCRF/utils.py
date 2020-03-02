import time
import torch
import numpy as np
from torch import nn


def classListToTensor(sentence_classes):
    tensor = torch.zeros((1, len(sentence_classes)), dtype=torch.long)
    for idx, label in enumerate(sentence_classes):
        tensor[0, idx] = label
    return tensor


def value_to_key(in_value, entity_label_dict):
    for key, dict_value in entity_label_dict.items():
        if in_value == dict_value:
            return key


def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    current_time = time.asctime(time.localtime())
    text = "Progress: [{0}] {1:.1f}% | Time: {2}".format( "#" * block + "-" * (bar_length - block), progress * 100, current_time)
    print(text)


r""" This method does not ensure that the whole dataset is used, it just uses random numbers the whole time to sample """
def generate_batch(tokenized_sentences, sentences_embeddings, embedding_dimension, classes, batch_size, device):

    batch_idx = np.random.randint(low=0,high=len(tokenized_sentences), size=batch_size).tolist()
    batch_tokenized_sentences = [tokenized_sentences[i] for i in batch_idx]
    batch_classes = [torch.LongTensor(classes[i]) for i in batch_idx]

    sentences_length = torch.LongTensor([len(sentence) for sentence in batch_tokenized_sentences])
    sentences_tensor = torch.zeros((batch_size, sentences_length.max(), embedding_dimension), dtype=torch.float).to(device)
    mask = torch.zeros((batch_size, sentences_length.max()), dtype=torch.long).to(device)

    for idx, (sentence_idx, sentence_length) in enumerate(zip(batch_idx, sentences_length)):
        # #seq_vec = train_emb[sentence_idx][0][2]
        # #seq_vec = torch.cat((train_emb[sentence_idx][0][0],train_emb[sentence_idx][0][2]),dim=1)
        # seq_vec = train_emb[sentence_idx][0].mean(dim=0)
        r""" Embeddings in numpy array must be first converted to torch tensor """
        sentence_tensor = torch.from_numpy(sentences_embeddings[sentence_idx, :sentence_length * embedding_dimension]).float()
        sentence_tensor = sentence_tensor.view(sentence_length, embedding_dimension)
        sentences_tensor[idx, :sentence_length] = sentence_tensor
        mask[idx, :sentence_length] = 1

    sorted_len_units, perm_idx = sentences_length.sort(0, descending=True)
    sentences_tensor = sentences_tensor[perm_idx]
    mask = mask[perm_idx]

    sorted_batch_classes = []
    for idx in perm_idx:
        sorted_batch_classes.append(batch_classes[idx])

    packed_input = nn.utils.rnn.pack_padded_sequence(sentences_tensor, sorted_len_units, batch_first=True)

    return sentences_tensor, sorted_batch_classes, sorted_len_units, packed_input, mask

def loadModelConfigs(settings):
    class Args:
        pass
    configs = Args()
    configs.ENTITY_PREDICTION = settings["DLmodelparams"]["entity_prediction"]
    configs.epochs = int(settings["DLmodelparams"]["epochs"])
    configs.iterations_per_epoch = int(settings["DLmodelparams"]["iterationsperepoch"])
    configs.hidden_size = int(settings["DLmodelparams"]["hiddensize"])
    configs.batch_size = int(settings["DLmodelparams"]["batchsize"])
    configs.num_layers = int(settings["DLmodelparams"]["numlayers"])
    configs.learning_rate = float(settings["DLmodelparams"]["learningrate"])
    configs.WORDVEC_SIZE = int(settings["embeddings"]["wordvec_size"])
    return configs

def task1PredictionToOutput(modelPrediction, singleTokenizedDocument):
    observationsList = list()
    familyMemberList = list()
    observation = ""
    for idx, prediction in enumerate(modelPrediction):
        if prediction != 0:
            if prediction == 1:
                observation = singleTokenizedDocument[idx]
            elif prediction == 2:
                observation = observation + " " + singleTokenizedDocument[idx]
                if idx < len(modelPrediction):
                    if modelPrediction[idx + 1] != 2:
                        observationsList.append(observation)
            elif prediction in (3, 4, 5):
                # Trim plurals
                if singleTokenizedDocument[idx][-1] == "s":
                    familyMember = singleTokenizedDocument[idx][:-1].capitalize()
                else:
                    familyMember = singleTokenizedDocument[idx].capitalize()
                if prediction == 3:
                    familyMemberList.append(tuple((familyMember, "Paternal")))
                elif prediction == 4:
                    familyMemberList.append(tuple((familyMember, "Maternal")))
                elif prediction == 5:
                    familyMemberList.append(tuple((familyMember, "NA")))
    # Set is used to remove duplicate entries
    familyMemberList = list(set(familyMemberList))
    observationsList = list(set(observationsList))
    return familyMemberList, observationsList


