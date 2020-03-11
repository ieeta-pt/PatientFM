import torch
import numpy as np

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
    configs.EMBEDDINGS_FREEZE_AFTER_EPOCH = int(settings["DLmodelparams"]["EMBEDDINGS_FREEZE_AFTER_EPOCH"])
    return configs


r""" This method does not ensure that the whole dataset is used, it just uses random numbers the whole time to sample """
def generateBatch(tokenized_sentences_tensors, classes, batch_size, device):

    batch_idx = np.random.randint(low=0,high=len(tokenized_sentences_tensors), size=batch_size).tolist()
    batch_tokenized_sentences_tensors = [tokenized_sentences_tensors[i] for i in batch_idx]
    batch_classes = [torch.LongTensor(classes[i]) for i in batch_idx]

    sentences_length = torch.LongTensor([sentence_tensor.shape[0] for sentence_tensor in batch_tokenized_sentences_tensors])
    sentences_tensor = torch.zeros((batch_size, sentences_length.max()), dtype=torch.long).to(device)
    mask = torch.zeros((batch_size, sentences_length.max()), dtype=torch.long).to(device)

    for idx, (sentence_idx, sentence_length) in enumerate(zip(batch_idx, sentences_length)):
        sentences_tensor[idx, :sentence_length] = batch_tokenized_sentences_tensors[idx]
        mask[idx, :sentence_length] = 1

    sorted_len_units, perm_idx = sentences_length.sort(0, descending=True)
    sentences_tensor = sentences_tensor[perm_idx]
    mask = mask[perm_idx]

    sorted_batch_classes = []
    for idx in perm_idx:
        sorted_batch_classes.append(batch_classes[idx])

    return sentences_tensor, sorted_batch_classes, sorted_len_units, mask

#
# def createTrainOutputTask1(DLmodel, testTokenizedSentences, testEncodedSentences, testClasses, testDocMapping):
#     """
#     Runs the trained model on the validation split, returning the resulting entity predictions
#     :param DLmodel:
#     :param testTokenizedSentences:
#     :param testEncodedSentences:
#     :param testClasses:
#     :param testDocMapping:
#     :return:
#     """
#
#     predFamilyMemberDict = {}
#     predObservationDict = {}
#     for idx, _ in enumerate(testTokenizedSentences):
#         testModelPred, _ = DLmodel.test([testEncodedSentences[idx]], testClasses[idx], SINGLE_INSTANCE=True)
#         familyMemberList, observationsList = trainPredictionToOutputTask1(testModelPred, testTokenizedSentences[idx])
#         if familyMemberList:
#             if testDocMapping[idx] not in predFamilyMemberDict.keys():
#                 predFamilyMemberDict[testDocMapping[idx]] = []
#             predFamilyMemberDict[testDocMapping[idx]].extend(familyMemberList)
#         if observationsList:
#             if testDocMapping[idx] not in predObservationDict.keys():
#                 predObservationDict[testDocMapping[idx]] = []
#             predObservationDict[testDocMapping[idx]].extend(observationsList)
#     return predFamilyMemberDict, predObservationDict


# def testPredictionToOutputTask1(modelPrediction, singleTokenizedDocument):
#     """
#     Converts the prediction vector to the respective entities identified in the text
#     :param modelPrediction:
#     :param singleTokenizedDocument:
#     :return:
#     """
#     observationsList = list()
#     familyMemberList = list()
#     observation = ""
#     finalCheckablePosition = len(modelPrediction) - 1
#     for idx, prediction in enumerate(modelPrediction):
#         if prediction != 0:
#             if prediction in (1, 2):
#                 if prediction == 1:
#                     observation = singleTokenizedDocument[idx]
#                 elif prediction == 2:
#                     observation = observation + " " + singleTokenizedDocument[idx]
#                 if idx < finalCheckablePosition:
#                     if modelPrediction[idx + 1] not in (1, 2):
#                         observationsList.append(observation)
#                         observation = ""
#             elif prediction in (3, 4, 5):
#                 # Trim plurals
#                 if singleTokenizedDocument[idx][-1] == "s":
#                     familyMember = singleTokenizedDocument[idx][:-1].capitalize()
#                 else:
#                     familyMember = singleTokenizedDocument[idx].capitalize()
#                 if prediction == 3:
#                     familyMemberList.append(tuple((familyMember, "Paternal")))
#                 elif prediction == 4:
#                     familyMemberList.append(tuple((familyMember, "Maternal")))
#                 elif prediction == 5:
#                     familyMemberList.append(tuple((familyMember, "NA")))
#     # Set is used to remove duplicate entries
#     familyMemberList = list(set(familyMemberList))
#     observationsList = list(set(observationsList))
#     return familyMemberList, observationsList
