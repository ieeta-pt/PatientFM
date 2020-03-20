import torch
import numpy as np

from Preprocessing import nltkSentenceSplit, nltkTokenize

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
    configs.USE_NEJI = settings["neji"]["use_neji_annotations"]
    configs.ALBERT_MODEL = settings["ALBERT"]["model"]
    return configs

def getSentenceListWithMapping(filesRead, AlbertTokenizer, device):
    """
    Returns a list of LongTensors with tokenized+encoded sentences for the ALBERT model
    :param filesRead:
    :param AlbertTokenizer:
    :return:
    """
    sentenceList = list()
    sentenceToDocList = list()
    encodedTokenizedSentenceList = list()
    for fileName in filesRead:
        sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
        sentenceToDocList.extend(fileName for _ in sentences)
        for sentence in sentences:
            nltkTokenizedSentence = nltkTokenize(sentence)
            print(nltkTokenizedSentence)
            nltkTokenizedSentence.insert(0, " ")
            nltkTokenizedSentence.insert(-1, " ")
            print(nltkTokenizedSentence)

            sentence = AlbertTokenizer.encode(sentence, add_special_tokens=True)
            sentence = torch.LongTensor(sentence).to(device=device)
            encodedTokenizedSentenceList.append(sentence)

    return encodedTokenizedSentenceList, sentenceToDocList



#to arrange all methods below

def generateBatch(tokenized_sentences_tensors, classes, batch_size, device, neji_classes=None):

    batch_idx = np.random.randint(low=0,high=len(tokenized_sentences_tensors), size=batch_size).tolist()
    batch_tokenized_sentences_tensors = [tokenized_sentences_tensors[i] for i in batch_idx]
    batch_classes = [torch.LongTensor(classes[i]) for i in batch_idx]

    sentences_length = torch.LongTensor([sentence_tensor.shape[0] for sentence_tensor in batch_tokenized_sentences_tensors])
    sentences_tensor = torch.zeros((batch_size, sentences_length.max()), dtype=torch.long).to(device)
    mask = torch.zeros((batch_size, sentences_length.max()), dtype=torch.long).to(device)

    if neji_classes is not None:
        batch_neji_classes = [torch.FloatTensor(neji_classes[i]) for i in batch_idx]
        neji_padded_classes = torch.zeros((batch_size, sentences_length.max()), dtype=torch.float).to(device)

    for idx, (sentence_idx, sentence_length) in enumerate(zip(batch_idx, sentences_length)):
        sentences_tensor[idx, :sentence_length] = batch_tokenized_sentences_tensors[idx]
        if neji_classes is not None:
            neji_padded_classes[idx, :sentence_length] = batch_neji_classes[idx]
        mask[idx, :sentence_length] = 1

    sorted_len_units, perm_idx = sentences_length.sort(0, descending=True)
    sentences_tensor = sentences_tensor[perm_idx]
    mask = mask[perm_idx]

    sorted_batch_classes = []
    for idx in perm_idx:
        sorted_batch_classes.append(batch_classes[idx])

    if neji_classes is not None:
        neji_padded_classes = neji_padded_classes[perm_idx]

        return sentences_tensor, sorted_batch_classes, sorted_len_units, mask, neji_padded_classes

    return sentences_tensor, sorted_batch_classes, sorted_len_units, mask


def concatenateNejiClassesToEmbeddings(embeddings, nejiClasses, device):
    """
    Appends neji classes to embeddings, to add novel information to model input
    :param embeddings: torch tensor with embeddings for each sentence
    :param nejiClasses: torch tensors with neji classes for each sentence
    :param device:
    :return:
    """
    nejiTensor = nejiClasses.unsqueeze(2).to(device)
    newTensor = torch.cat((embeddings, nejiTensor), dim=2)
    return newTensor
