import torch
import numpy as np
from transformers import AlbertTokenizer

from Entity import filterObservations, filterFamilyMembers, FAMILY_PLURALS, ENTITY_CLASSES
from Preprocessing import nltkSentenceSplit, nltkTokenize

ALBERT_ENTITY_CLASSES = {"O": 0,
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
    configs.ALBERT_MODEL = settings["ALBERT"]["model"]
    return configs

class ALBERTutils():
    def __init__(self, albert_model, add_special_tokens=True):
        self.addSpecialTokens = add_special_tokens
        self.albertTokenizer = AlbertTokenizer.from_pretrained(albert_model)
        self.albertEntityClasses = ALBERT_ENTITY_CLASSES

    def getSentenceListWithMapping(self, filesRead):
        """
        Returns a list with tokenized+encoded sentences for the ALBERT model
        :param filesRead:
        :return sentenceList: list of nltkTokenized sentences
        :return encodedTokenizedSentenceList: list of ALBERT tokenized and encoded sentences
        :return sentenceToDocList: sentence to doc mapping
        """
        sentenceList = list()
        sentenceToDocList = list()
        encodedTokenizedSentenceList = list()
        for fileName in filesRead:
            sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
            sentenceToDocList.extend(fileName for _ in sentences)
            for sentence in sentences:
                nltkTokenizedSentence = nltkTokenize(sentence)
                sentenceList.append(nltkTokenizedSentence)

                sentence = self.albertTokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                encodedTokenizedSentenceList.append(sentence)
        return sentenceList, encodedTokenizedSentenceList, sentenceToDocList

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
                sentence = self.albertTokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
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
                observationList = [self.albertTokenizer.tokenize(observation[0]) for _, observation in observations] #The [0] is to store the string instead of a list with a string
                observationIdx = 0
                for sentencePosition, sentence in enumerate(sentences):
                    tokenIdx = 0
                    sentence = self.albertTokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                    sentence = self.albertTokenizer.convert_ids_to_tokens(sentence)
                    for tokenPosition, token in enumerate(sentence):
                        if observationIdx <= len(observationList) - 1:
                            if token == observationList[observationIdx][tokenIdx]:
                                if tokenIdx == 0:
                                    classesDict[fileName][sentencePosition][tokenPosition] = self.albertEntityClasses["B-Observation"]
                                    if len(observationList[observationIdx]) == 1:
                                        observationIdx += 1
                                    else:
                                        tokenIdx += 1
                                else:
                                    classesDict[fileName][sentencePosition][tokenPosition] = self.albertEntityClasses["I-Observation"]
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
                familyMemberList = [(self.albertTokenizer.tokenize(familyMember.lower()), familySide) for _, familyMember, familySide in familyMembers]
                familyMemberIdx = 0
                for sentencePosition, sentence in enumerate(sentences):
                    tokenIdx = 0
                    sentence = self.albertTokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
                    sentence = self.albertTokenizer.convert_ids_to_tokens(sentence)
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

                                elif familyMemberList[familyMemberIdx][0][tokenIdx] == "▁grand":
                                    if token.startswith("▁grand"):
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

    # def setFamilyMemberClasses(self, classesDict, datasetTXT, datasetXML):
    #     """
    #     This method updates default classes with the classes for FamilyMember entities
    #
    #     :param classesDict: dict with key:filename, value:list of lists with default classes per sentence in the document
    #     :param datasetTXT: dict with text from txt files indexed by filename
    #     :param datasetXML: dict with annotations from xml files indexed by filename
    #     :return: Dict with key:filename, value:list of lists with classes per sentence in the document
    #     """
    #     for fileName in datasetTXT:
    #         sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
    #         familyMembers = filterFamilyMembers(datasetXML[fileName])
    #         # Some documents can have no family member mentions. In that case, the class list remains unchanged
    #         if familyMembers:
    #             familyMemberList = [(self.albertTokenizer.tokenize(familyMember.lower()), familySide) for _, familyMember, familySide in familyMembers]
    #             familyMemberIdx = 0
    #             for sentencePosition, sentence in enumerate(sentences):
    #                 tokenIdx = 0
    #                 sentence = self.albertTokenizer.encode(sentence, add_special_tokens=self.addSpecialTokens)
    #                 sentence = self.albertTokenizer.convert_ids_to_tokens(sentence)
    #                 matchFound = False
    #                 beginToken = True
    #
    #                 #                sentence = ["_yes","_this","_sentence","_daug","ter", "_son", "_uh", "_eh", "_zing","_child","s","_son", "."]
    #                 #                classesDict[fileName][sentencePosition] = [0,0,0,0,0,0,0,0,0,0,0,0,0] #[0,0,0,3,4,7,0,0,0,7,8,0]
    #                 #                print(sentence)
    #                 #                familyMemberList = [(["_daug", "ter"],"Paternal"),(["_son"], "NA"), (["_child","s"],"NA"), (["_son"], "NA")]
    #
    #                 for tokenPosition, token in enumerate(sentence):
    #                     #                    print(classesDict[fileName][sentencePosition])
    #                     if familyMemberIdx <= len(familyMemberList) - 1:
    #                         #                        print(familyMemberList[familyMemberIdx][0][tokenIdx], token)
    #                         if len(familyMemberList[familyMemberIdx][0]) == 1:
    #                             #                            print("In single member")
    #                             if token == familyMemberList[familyMemberIdx][0][tokenIdx]:
    #                                 classesDict[fileName][sentencePosition][tokenPosition] = \
    #                                     self.getFamilyClass(familyMemberList[familyMemberIdx][1], beginningToken=beginToken)
    #                                 familyMemberIdx += 1
    #
    #                         elif len(familyMemberList[familyMemberIdx][0]) > 1 and not matchFound:
    #                             #                            print("In multiple member")
    #                             #We must check if it is a full match before assigning classes
    #                             if token == familyMemberList[familyMemberIdx][0][tokenIdx]:
    #                                 maxOffset = len(familyMemberList[familyMemberIdx][0])-1
    #                                 #to check if we are not going to check out of boundaries
    #                                 if tokenPosition+maxOffset-tokenIdx <= len(sentence)-1:
    #                                     matchFound = True
    #                                     beginToken = True
    #                                     for idx, mention in enumerate(familyMemberList[familyMemberIdx][0]):
    #                                         if sentence[tokenPosition+idx] != mention:
    #                                             matchFound = False
    #                         if matchFound:
    #                             #                            print("Matched {} with {}".format(familyMemberList[familyMemberIdx][0][tokenIdx],token))
    #                             classesDict[fileName][sentencePosition][tokenPosition] = \
    #                                 self.getFamilyClass(familyMemberList[familyMemberIdx][1], beginningToken=beginToken)
    #                             beginToken = False
    #                             tokenIdx += 1
    #                             if tokenIdx == len(familyMemberList[familyMemberIdx][0]):
    #                                 #                                print("finished match")
    #                                 tokenIdx = 0
    #                                 familyMemberIdx += 1
    #                                 beginToken = True
    #                                 matchFound = False
    #
    #     return classesDict



    def convertClassesForALBERT(self, nltkTokenizedSentences, albertEncodedSentences, classesList):
        """
        Converts classes for the nltk tokenized sentences to albert tokenized sentences
        :param nltkTokenizedSentences:
        :param albertEncodedSentences:
        :param classesList: list of classes for the nltkTokenizedSentences
        :param add_special_tokens: used to check if it is necessary to pad classes with a dummy class in begin and end
        :return: list of classes for the albertEncodedSentences
        """
        assert len(nltkTokenizedSentences) == len(albertEncodedSentences) == len(classesList), "All lists must have the same length"

        albertClasses = list()
        for nltkSentence, albertSentence, nltkClasses in zip(nltkTokenizedSentences, albertEncodedSentences, classesList):
            if self.addSpecialTokens:
                albertSentenceClasses = [self.albertEntityClasses["O"]]
                albertSentence = self.albertTokenizer.convert_ids_to_tokens(albertSentence, skip_special_tokens=True)
            else:
                albertSentenceClasses = []
                albertSentence = self.albertTokenizer.convert_ids_to_tokens(albertSentence)

            print(self.albertTokenizer.tokenize("shit this doesn't make sense"))
            print(nltkTokenize("shit this doesn't make sense"))

    #         nltkSentence = ['the', 'patient', "'s", 'paternal', 'grandfather', 'was', 'diagnosed', 'with', 'ear', 'melanoma', 'at', 'age', '94', 'and', 'passed', 'away', 'at', 'age', '95', '.']
    #         albertSentence = ['▁the', '▁patient', "'", 's', '▁paternal', '▁grandfather', '▁was', '▁diagnosed', '▁with', '▁ear', '▁melano', 'ma', '▁at', '▁age', '▁94', '▁and', '▁passed', '▁away', '▁at', '▁age', '▁95', '.']
    #         print("\n")
    #         print(nltkSentence)
    #         print(albertSentence)
    #
    #
    #         albertTokenIdx = 0
    #         albertLastTokenIdx = len(albertSentence) - 1
    #         for nltkIdx, nltkToken in enumerate(nltkSentence):
    #             goToNextNLTKToken = False
    #             while not goToNextNLTKToken:
    #                 albertToken = albertSentence[albertTokenIdx]
    #                 if nltkToken == "’":
    #                     print("fuckthishitttt")
    #                 print(albertTokenIdx, len(albertSentence)-1)
    #                 print(nltkIdx, len(nltkSentence)-1)
    #                 print(nltkToken)
    #                 print(albertToken)
    #                 if albertToken.startswith('▁'):
    #                     savedAlbertToken = albertToken.strip('▁')
    #                     albertSentenceClasses.append(self.getAlbertClass(nltkClasses[nltkIdx], beginningToken=True))
    #                     albertTokenIdx += 1
    #                     if savedAlbertToken == nltkToken:
    #                         goToNextNLTKToken = True
    #
    #                 # elif albertToken == '.' and albertTokenIdx == albertLastTokenIdx:
    #                 #     albertSentenceClasses.append(self.getAlbertClass(nltkClasses[nltkIdx], beginningToken=False))
    #                 #     goToNextNLTKToken = True
    #
    #                 elif albertToken.startswith('\''):
    #                     savedAlbertToken += albertToken
    #                     albertSentenceClasses.append(self.getAlbertClass(nltkClasses[nltkIdx], beginningToken=False))
    #                     if savedAlbertToken == nltkToken:
    #                         goToNextNLTKToken = True
    #
    #                 else:
    #                     savedAlbertToken += albertToken
    #                     albertSentenceClasses.append(self.getAlbertClass(nltkClasses[nltkIdx], beginningToken=False))
    #                     albertTokenIdx += 1
    #                     if savedAlbertToken == nltkToken:
    #                         goToNextNLTKToken = True
    #
    #                 if goToNextNLTKToken:
    #                     savedAlbertToken = ""
    #
    #         albertClasses.append(albertSentenceClasses)
    #
    #
    #         print(nltkSentence)
    #         print(nltkClasses)
    #         print(albertSentence)
    #         print(albertSentenceClasses)
    #         print("\n")
    # #        print(nltkClasses)


        return

    def getFamilyClass(self, familySide, beginningToken=True):

        if familySide == "Paternal":
            if beginningToken: return self.albertEntityClasses["BP-FamilyMember"]
            else:              return self.albertEntityClasses["IP-FamilyMember"]

        elif familySide == "Maternal":
            if beginningToken: return self.albertEntityClasses["BM-FamilyMember"]
            else:              return self.albertEntityClasses["IM-FamilyMember"]

        elif familySide == "NA":
            if beginningToken: return self.albertEntityClasses["BNA-FamilyMember"]
            else:              return self.albertEntityClasses["INA-FamilyMember"]



# sentence = 'A daughter, age 25, yes iii.5token ii,5tokentoo, ae and a health.'
# print(nltkTokenize(sentence))
# print(AlbertTokenizer.tokenize(sentence))
#print(AlbertTokenizer.convert_ids_to_tokens(sentence))

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
