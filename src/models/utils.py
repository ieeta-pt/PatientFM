import time
import torch
from torch import nn
import numpy as np

from Preprocessing import nltkSentenceSplit, nltkTokenize


def valueToKey(in_value, entity_label_dict):
    for key, dict_value in entity_label_dict.items():
        if in_value == dict_value:
            return key

def updateProgress(progress):
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


def classDictToList(classDict):
    classList = list()
    for fileName in classDict:
        classList.extend(sentenceClasses for sentenceClasses in classDict[fileName])
    return classList


def classListToTensor(sentence_classes, datatype):
    tensor = torch.zeros((1, len(sentence_classes)), dtype=datatype)
    for idx, label in enumerate(sentence_classes):
        tensor[0, idx] = label
    return tensor

def getSentenceList(filesRead, tokenized=False):
    sentenceList = list()
    for fileName in filesRead:
        sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
        if tokenized:
            sentenceList.extend(nltkTokenize(sentence) for sentence in sentences)
        else:
            sentenceList.extend(sentence for sentence in sentences)
    return sentenceList


def getSentenceListWithMapping(filesRead, tokenized=False):
    sentenceToDocList = list()
    sentenceList = list()
    for fileName in filesRead:
        sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
        sentenceToDocList.extend(fileName for _ in sentences)
        if tokenized:
            sentenceList.extend(nltkTokenize(sentence) for sentence in sentences)
        else:
            sentenceList.extend(sentence for sentence in sentences)
    return sentenceList, sentenceToDocList


r""" This method does not ensure that the whole dataset is used, it just uses random numbers the whole time to sample """
def generateBatch(tokenized_sentences, sentences_embeddings, embedding_dimension, classes, batch_size, device):

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


def createOutputTask1(DLmodel, testTokenizedSentences, testEncodedSentences, testClasses, testDocMapping, neji_classes=None):
    """
    Runs the trained model on the unseen data split (validation or test), returning the resulting entity predictions
    :param DLmodel:
    :param testTokenizedSentences:
    :param testEncodedSentences:
    :param testClasses:
    :param testDocMapping:
    :param neji_classes: neji classes in case neji annotations are used to add input information to the model
    :return:
    """

    predFamilyMemberDict = {}
    predObservationDict = {}
    for idx, _ in enumerate(testTokenizedSentences):
        if neji_classes is not None:
            nejiClasses = neji_classes[idx]
        else:
            nejiClasses = None

        testModelPred, _ = DLmodel.test([testEncodedSentences[idx]], testClasses[idx], SINGLE_INSTANCE=True, neji_classes=nejiClasses)
        familyMemberList, observationsList = predictionToOutputTask1(testModelPred, testTokenizedSentences[idx])
        if familyMemberList:
            if testDocMapping[idx] not in predFamilyMemberDict.keys():
                predFamilyMemberDict[testDocMapping[idx]] = []
            predFamilyMemberDict[testDocMapping[idx]].extend(familyMemberList)
        if observationsList:
            if testDocMapping[idx] not in predObservationDict.keys():
                predObservationDict[testDocMapping[idx]] = []
            predObservationDict[testDocMapping[idx]].extend(observationsList)
    return predFamilyMemberDict, predObservationDict


def predictionToOutputTask1(modelPrediction, singleTokenizedSentence):
    """
    Converts the prediction vector to the respective entities identified in the text
    :param modelPrediction:
    :param singleTokenizedDocument:
    :return:
    """
    observationsList = list()
    familyMemberList = list()
    observation = ""
    finalCheckablePosition = len(modelPrediction) - 1
    for idx, prediction in enumerate(modelPrediction):
        if prediction != 0:
            if prediction in (1, 2):
                if prediction == 1:
                    observation = singleTokenizedSentence[idx]
                elif prediction == 2:
                    observation = observation + " " + singleTokenizedSentence[idx]
                if idx < finalCheckablePosition:
                    if modelPrediction[idx + 1] not in (1, 2):
                        observationsList.append(observation)
                        observation = ""
            elif prediction in (3, 4, 5):
                # Trim plurals
                if singleTokenizedSentence[idx][-1] == "s":
                    familyMember = singleTokenizedSentence[idx][:-1].capitalize()
                else:
                    familyMember = singleTokenizedSentence[idx].capitalize()
                if prediction == 3:
                    familyMemberList.append(tuple((familyMember, "Paternal")))
                elif prediction == 4:
                    familyMemberList.append(tuple((familyMember, "Maternal")))
                elif prediction == 5:
                    familyMemberList.append(tuple((familyMember, "NA")))
    return familyMemberList, observationsList


def mergeDictionaries(dictionaries):
    dictionary = {}
    for d in dictionaries:
        for key, value in d.items():
            if key not in dictionary.keys():
                dictionary[key] = []
            dictionary[key].extend(value)
    for key in dictionary:
        dictionary[key] = list(set(dictionary[key]))
    return dictionary
