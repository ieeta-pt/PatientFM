import torch
import numpy as np
from transformers import AutoTokenizer

from Entity import filterObservations, filterFamilyMembers, FAMILY_PLURALS, ENTITY_CLASSES
from Preprocessing import nltkSentenceSplit

BERT_ENTITY_CLASSES = {"O": 0,
                         "B-Observation": 1,  "I-Observation": 2,
                         "BP-FamilyMember": 3, "IP-FamilyMember": 4,
                         "BM-FamilyMember": 5, "IM-FamilyMember": 6,
                         "BNA-FamilyMember": 7, "INA-FamilyMember": 8}

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
    configs.patience = int(settings["DLmodelparams"]["patience"])
    return configs

class clinicalBERTutils():
    def __init__(self, add_special_tokens=True):
        self.addSpecialTokens = add_special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.BERTEntityClasses = BERT_ENTITY_CLASSES

    def getSentenceListWithMapping(self, filesRead):
        """
        Returns a list with tokenized+encoded sentences for the ALBERT model
        :param filesRead:
        :return tokenizedSentenceList: list of ALBERT tokenized sentences
        :return encodedTokenizedSentenceList: list of ALBERT tokenized and encoded sentences
        :return sentenceToDocList: sentence to doc mapping
        """
        sentenceToDocList = list()
        tokenizedSentenceList = list()
        encodedTokenizedSentenceList = list()
        for fileName in filesRead:
            sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
            sentenceToDocList.extend(fileName for _ in sentences)
            for sentence in sentences:
                sentence = self.tokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                encodedTokenizedSentenceList.append(sentence)
                sentence = self.tokenizer.convert_ids_to_tokens(sentence)
                tokenizedSentenceList.append(sentence)
        return tokenizedSentenceList, encodedTokenizedSentenceList, sentenceToDocList

    def createDefaultClasses(self, datasetTXT):
        """
        :param datasetTXT: dict with text from txt files indexed by filename
        :return: Dict with key:filename, value:list of lists with classes per sentence in the document
        """
        classesDict = {}
        for fileName in datasetTXT:
            classesDict[fileName] = []
            sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
            for sentence in sentences:
                sentence = self.tokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                classesDict[fileName].append([int(0) for _ in sentence])
        return classesDict

    def createTrueClasses(self, datasetTXT, XMLAnnotations):
        """
        Creates default classes and populates them with the golden standard from the annotations (Observations and FamilyMember annotations only)
        :param datasetTXT:
        :param XMLAnnotations:
        :return: classesDict
        """
        classesDict = self.createDefaultClasses(datasetTXT)
        classesDict = self.setObservationClasses(classesDict, datasetTXT, XMLAnnotations)
        classesDict = self.setFamilyMemberClasses(classesDict, datasetTXT, XMLAnnotations)
        return classesDict

    def setObservationClasses(self, classesDict, datasetTXT, datasetXML):
        """
        This method updates default classes with the classes for Observation entities

        :param classesDict: dict with key:filename, value:list of lists with default classes per sentence in the document
        :param datasetTXT: dict with text from txt files indexed by filename
        :param datasetXML: dict with annotations from xml files indexed by filename
        :return: Dict with key:filename, value:list of lists with classes per sentence in the document
        """
        for fileName in datasetTXT:
            sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
            observations = filterObservations(datasetXML[fileName])
            # Some documents can have no observations. In that case, the default class list remains unchanged
            if observations:
                observationList = [self.tokenizer.tokenize(observation[0]) for _, observation in observations] #The [0] is to store the string instead of a list with a string
                observationIdx = 0
                for sentencePosition, sentence in enumerate(sentences):
                    tokenIdx = 0
                    sentence = self.tokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                    sentence = self.tokenizer.convert_ids_to_tokens(sentence)
                    for tokenPosition, token in enumerate(sentence):
                        if observationIdx <= len(observationList) - 1:
                            if token == observationList[observationIdx][tokenIdx]:
                                if tokenIdx == 0:
                                    classesDict[fileName][sentencePosition][tokenPosition] = self.BERTEntityClasses["B-Observation"]
                                    if len(observationList[observationIdx]) == 1:
                                        observationIdx += 1
                                    else:
                                        tokenIdx += 1
                                else:
                                    classesDict[fileName][sentencePosition][tokenPosition] = self.BERTEntityClasses["I-Observation"]
                                    tokenIdx += 1
                                    if tokenIdx == len(observationList[observationIdx]):
                                        tokenIdx = 0
                                        observationIdx += 1
        return classesDict

    def setFamilyMemberClasses(self, classesDict, datasetTXT, datasetXML):
        """
        This method updates default classes with the classes for FamilyMember entities

        :param classesDict: dict with key:filename, value:list of lists with default classes per sentence in the document
        :param datasetTXT: dict with text from txt files indexed by filename
        :param datasetXML: dict with annotations from xml files indexed by filename
        :return: Dict with key:filename, value:list of lists with classes per sentence in the document
        """
        for fileName in datasetTXT:
            sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
            familyMembers = filterFamilyMembers(datasetXML[fileName])
            # Some documents can have no family member mentions. In that case, the class list remains unchanged
            if familyMembers:
                familyMemberList = [(self.tokenizer.tokenize(familyMember.lower()), familySide) for _, familyMember, familySide in familyMembers]
                familyMemberIdx = 0
                for sentencePosition, sentence in enumerate(sentences):
                    tokenIdx = 0
                    sentence = self.tokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                    sentence = self.tokenizer.convert_ids_to_tokens(sentence)
                    matchFound = False
                    beginToken = True
                    for tokenPosition, token in enumerate(sentence):
                        if familyMemberIdx <= len(familyMemberList) - 1:
                            if len(familyMemberList[familyMemberIdx][0]) == 1:
                                if token == familyMemberList[familyMemberIdx][0][tokenIdx]:
                                    classesDict[fileName][sentencePosition][tokenPosition] =\
                                       self.getFamilyClass(familyMemberList[familyMemberIdx][1], beginningToken=beginToken)
                                    familyMemberIdx += 1

                            elif len(familyMemberList[familyMemberIdx][0]) > 1 and not matchFound:
                                #We must check if it is a full match before assigning classes
                                if token == familyMemberList[familyMemberIdx][0][tokenIdx]:
                                    maxOffset = len(familyMemberList[familyMemberIdx][0])-1
                                    #to check if we are not going to check out of boundaries
                                    if tokenPosition+maxOffset-tokenIdx <= len(sentence)-1:
                                        matchFound = True
                                        beginToken = True
                                        for idx, mention in enumerate(familyMemberList[familyMemberIdx][0]):
                                            if sentence[tokenPosition+idx] != mention:
                                                matchFound = False

                                elif familyMemberList[familyMemberIdx][0][tokenIdx] == "grand":
                                    if token.startswith("grand"):
                                        classesDict[fileName][sentencePosition][tokenPosition] = \
                                            self.getFamilyClass(familyMemberList[familyMemberIdx][1], beginningToken=beginToken)
                                        familyMemberIdx += 1

                            if matchFound:
                                classesDict[fileName][sentencePosition][tokenPosition] = \
                                    self.getFamilyClass(familyMemberList[familyMemberIdx][1], beginningToken=beginToken)
                                beginToken = False
                                tokenIdx += 1
                                if tokenIdx == len(familyMemberList[familyMemberIdx][0]):
                                    tokenIdx = 0
                                    familyMemberIdx += 1
                                    beginToken = True
                                    matchFound = False
        return classesDict

    def getFamilyClass(self, familySide, beginningToken=True):

        if familySide == "Paternal":
            if beginningToken: return self.BERTEntityClasses["BP-FamilyMember"]
            else:              return self.BERTEntityClasses["IP-FamilyMember"]

        elif familySide == "Maternal":
            if beginningToken: return self.BERTEntityClasses["BM-FamilyMember"]
            else:              return self.BERTEntityClasses["IM-FamilyMember"]

        elif familySide == "NA":
            if beginningToken: return self.BERTEntityClasses["BNA-FamilyMember"]
            else:              return self.BERTEntityClasses["INA-FamilyMember"]



# def createOutputTask1(DLmodel, testTokenizedSentences, testEncodedSentences, testClasses, testDocMapping, bertUtils, neji_classes=None):
#     """
#     Runs the trained model on the unseen data split (validation or test), returning the resulting entity predictions
#     :param DLmodel:
#     :param testTokenizedSentences:
#     :param testEncodedSentences:
#     :param testClasses:
#     :param testDocMapping:
#     :param bertUtils: instance of BERTutils which contains the necessary tokenizer
#     :param neji_classes: neji classes in case neji annotations are used to add input information to the model
#     :return:
#     """
#
#     predFamilyMemberDict = {}
#     predObservationDict = {}
#     for idx, _ in enumerate(testTokenizedSentences):
#         if neji_classes is not None:
#             nejiClasses = neji_classes[idx]
#         else:
#             nejiClasses = None
#
#         testModelPred, _ = DLmodel.test([testEncodedSentences[idx]], testClasses[idx], SINGLE_INSTANCE=True, neji_classes=nejiClasses)
#         familyMemberList, observationsList = predictionToOutputTask1(testModelPred, testTokenizedSentences[idx], bertUtils)
#         if familyMemberList:
#             if testDocMapping[idx] not in predFamilyMemberDict.keys():
#                 predFamilyMemberDict[testDocMapping[idx]] = []
#             predFamilyMemberDict[testDocMapping[idx]].extend(familyMemberList)
#         if observationsList:
#             if testDocMapping[idx] not in predObservationDict.keys():
#                 predObservationDict[testDocMapping[idx]] = []
#             predObservationDict[testDocMapping[idx]].extend(observationsList)
#     return predFamilyMemberDict, predObservationDict


def createOutputTask1(DLmodel, testTokenizedSentences, testEncodedSentences, testClasses, testDocMapping, bertUtils, neji_classes=None):
    """
    Runs the trained model on the unseen data split (validation or test), returning the resulting entity predictions
    :param DLmodel:
    :param testTokenizedSentences:
    :param testEncodedSentences:
    :param testClasses:
    :param testDocMapping:
    :param bertUtils: instance of BERTutils which contains the necessary tokenizer
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

        reconstructedSentence = bertUtils.tokenizer.convert_tokens_to_string(testTokenizedSentences[idx])
        reconstructedSentence = reconstructedSentence.replace("[CLS] ", "").replace(" [SEP]", "")

        testModelPred, _ = DLmodel.test([testEncodedSentences[idx]], testClasses[idx], SINGLE_INSTANCE=True, neji_classes=nejiClasses)
        familyMemberList, observationsList = predictionToOutputTask1(testModelPred, testTokenizedSentences[idx], bertUtils)
        if familyMemberList:
            if testDocMapping[idx] not in predFamilyMemberDict.keys():
                predFamilyMemberDict[testDocMapping[idx]] = []
            for familyMember in familyMemberList:
                entry = tuple((familyMember, reconstructedSentence))
                predFamilyMemberDict[testDocMapping[idx]].extend([entry])

        if observationsList:
            if testDocMapping[idx] not in predObservationDict.keys():
                predObservationDict[testDocMapping[idx]] = []
            for observation in observationsList:
                entry = tuple((observation, reconstructedSentence))
                predObservationDict[testDocMapping[idx]].extend([entry])

    return predFamilyMemberDict, predObservationDict


def predictionToOutputTask1(modelPrediction, singleTokenizedSentence, bertUtils):
    """
    Converts the prediction vector to the respective entities identified in the text
    :param modelPrediction:
    :param singleTokenizedDocument:
    :param bertUtils: instance of clinicalBERTutils which contains the necessary tokenizer
    :return:
    """
    observationsList = list()
    familyMemberList = list()
    observation = list()
    familyMember = list()
    processFamilyMember = False
    finalCheckablePosition = len(modelPrediction) - 1
    for idx, prediction in enumerate(modelPrediction):
        if prediction != 0:
            if prediction in (1, 2):
                observation.append(singleTokenizedSentence[idx])
                if idx < finalCheckablePosition:
                    if modelPrediction[idx + 1] not in (1, 2):
                        #This if is used to "reconstruct" the beginning of the token in case a part of the token was not classified by the model
                        if observation[0].startswith("##"):
                            offset = len(observation)
                            while singleTokenizedSentence[idx-offset].startswith("##"):
                                observation.insert(0, singleTokenizedSentence[idx-offset])
                                offset += 1
                            observation.insert(0, singleTokenizedSentence[idx-offset]) #inserts beginning token
                        observationText = bertUtils.tokenizer.convert_tokens_to_string(observation)
                        if len(observationText) > 1:
                            observationsList.append(observationText)
                        observation = list()
            elif prediction in (3, 4):
                familyMember.append(singleTokenizedSentence[idx])
                if idx < finalCheckablePosition:
                    if modelPrediction[idx + 1] not in (3, 4):
                        familyMemberText = bertUtils.tokenizer.convert_tokens_to_string(familyMember)
                        familySide = "Paternal"
                        processFamilyMember = True
            elif prediction in (5, 6):
                familyMember.append(singleTokenizedSentence[idx])
                if idx < finalCheckablePosition:
                    if modelPrediction[idx + 1] not in (5, 6):
                        familyMemberText = bertUtils.tokenizer.convert_tokens_to_string(familyMember)
                        familySide = "Maternal"
                        processFamilyMember = True
            elif prediction in (7, 8):
                familyMember.append(singleTokenizedSentence[idx])
                if idx < finalCheckablePosition:
                    if modelPrediction[idx + 1] not in (7, 8):
                        familyMemberText = bertUtils.tokenizer.convert_tokens_to_string(familyMember)
                        familySide = "NA"
                        processFamilyMember = True

            if processFamilyMember:
                if len(familyMemberText) > 1:
                    if familyMemberText[-1] == "s":
                        familyMemberText = familyMemberText[:-1].capitalize()
                    else:
                        familyMemberText = familyMemberText.capitalize()
                    familyMemberList.append(tuple((familyMemberText, familySide)))
                familyMember = list()
                processFamilyMember = False

    return familyMemberList, observationsList
