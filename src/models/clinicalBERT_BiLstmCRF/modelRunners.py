import torch
import random
import numpy as np
from sklearn.model_selection import KFold

from Reader import Reader
from NejiAnnotator import readPickle

from models.utils import classListToTensor, classDictToList, getSentenceList, mergeDictionaries

from models.clinicalBERT.utils import BERT_ENTITY_CLASSES, loadModelConfigs, clinicalBERTutils, createOutputTask1
from models.clinicalBERT_BiLstmCRF.model import Model

def runModel(settings, trainTXT, trainXML):
    """ Trains the model in the FULL training dataset and computes predictions for the FULL test set
    :param settings: settings from settings.ini file
    :param trainTXT: train txts
    :param trainXML: train xml annotations
    :return: finalFamilyMemberDict, finalObservationsDict: dicts indexed by filename with detected entities
    """

    seed = [35899,54377,66449,77417,29,229,1229,88003,99901,11003]
    random_seed = seed[9]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


    print("Loading and preprocessing data.\n")

    if settings["ALBERT"]["add_special_tokens"] == "True":
        addSpecialTokens = True
    else:
        addSpecialTokens = False

    clinicalBERTUtils = clinicalBERTutils(addSpecialTokens)
    _, encodedTokenizedSentences, sentenceToDocList = clinicalBERTUtils.getSentenceListWithMapping(trainTXT)

    trainEncodedSentences = []
    for sentence in encodedTokenizedSentences:
        trainEncodedSentences.append(torch.LongTensor(sentence).to(device=device))

    trainClassesDict = clinicalBERTUtils.createTrueClasses(trainTXT, trainXML)
    trainClasses = classDictToList(trainClassesDict)
    trainClasses = [classListToTensor(sentenceClasses, datatype=torch.long) for sentenceClasses in trainClasses]

    if settings["neji"]["use_neji_annotations"] == "True":
        nejiClassesDict = readPickle(settings["neji"]["neji_train_pickle_clinicalbert"])
        nejiTrainClasses = classDictToList(nejiClassesDict)
        nejiTrainClasses = [classListToTensor(sentenceClasses, datatype=torch.float) for sentenceClasses in nejiTrainClasses]
    else:
        nejiTrainClasses = None

    # 100 is the default size used in embedding creation
    max_length = 100
    print("Loaded data successfully.\n")

    modelConfigs = loadModelConfigs(settings)

    DL_model = Model(modelConfigs, BERT_ENTITY_CLASSES, max_length, device)
    print("Model created. Starting training.\n")
    DL_model.train(trainEncodedSentences, trainClasses, neji_classes=nejiTrainClasses)

    print("Starting the testing phase.\n")
    reader = Reader(dataSettings=settings, corpus="test")
    testTXT = reader.loadDataSet()

    testBERTtokenizedSentences, encodedTokenizedSentences, sentenceToDocList = clinicalBERTUtils.getSentenceListWithMapping(testTXT)

    testEncodedSentences = []
    for sentence in encodedTokenizedSentences:
        testEncodedSentences.append(torch.LongTensor(sentence).to(device))

    testClassesDict = clinicalBERTUtils.createDefaultClasses(testTXT)
    testClasses = classDictToList(testClassesDict)
    testClasses = [classListToTensor(sentenceClasses, datatype=torch.long) for sentenceClasses in testClasses]

    if settings["neji"]["use_neji_annotations"] == "True":
        nejiTestClassesDict = readPickle(settings["neji"]["neji_test_pickle_clinicalbert"])
        nejiTestClasses = classDictToList(nejiTestClassesDict)
        nejiTestClasses = [classListToTensor(sentenceClasses, datatype=torch.float) for sentenceClasses in nejiTestClasses]
    else:
        nejiTestClasses = None

    predFamilyMemberDict, predObservationDict = createOutputTask1(DL_model, testBERTtokenizedSentences, testEncodedSentences,
                                                                  testClasses, sentenceToDocList, clinicalBERTUtils, neji_classes=nejiTestClasses)


    return predFamilyMemberDict, predObservationDict


def runModelDevelopment(settings, trainTXT, trainXML, cvFolds):
    """ Trains the model with K-fold cross validation, using K-1 splits to train and 1 to validate (and generate output), in K possible combinations.
    :param settings: settings from settings.ini file
    :param trainTXT: train txts
    :param trainXML: train xml annotations
    :param cvFolds: number of folds for cross validation
    :return: finalFamilyMemberDict, finalObservationsDict: dicts indexed by filename with detected entities
    """

    seed = [35899,54377,66449,77417,29,229,1229,88003,99901,11003]
    random_seed = seed[9]
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    print("Loading and preprocessing data.\n")

    if settings["ALBERT"]["add_special_tokens"] == "True":
        addSpecialTokens = True
    else:
        addSpecialTokens = False

    clinicalBERTUtils = clinicalBERTutils(addSpecialTokens)
    BERTtokenizedSentences, encodedTokenizedSentences, sentenceToDocList = clinicalBERTUtils.getSentenceListWithMapping(trainTXT)

    tensorEncodedSentences = []
    for sentence in encodedTokenizedSentences:
        tensorEncodedSentences.append(torch.LongTensor(sentence).to(device=device))

    classesDict = clinicalBERTUtils.createTrueClasses(trainTXT, trainXML)
    classes = classDictToList(classesDict)
    classes = [classListToTensor(sentenceClasses, datatype=torch.long) for sentenceClasses in classes]

    if settings["neji"]["use_neji_annotations"] == "True":
        nejiClassesDict = readPickle(settings["neji"]["neji_train_pickle_clinicalbert"])
        nejiClasses = classDictToList(nejiClassesDict)
        nejiClasses = [classListToTensor(sentenceClasses, datatype=torch.float) for sentenceClasses in nejiClasses]

    kFolds = KFold(n_splits=cvFolds)
    predFamilyMemberDicts = []
    predObservationsDicts = []

    print("Beginning KFold cross validation.\n")

    for trainIdx, testIdx in kFolds.split(encodedTokenizedSentences):

        trainEncodedSentences = [tensorEncodedSentences[idx] for idx in trainIdx]
        trainClasses = [classes[idx] for idx in trainIdx]

        testTokenizedSentences = [BERTtokenizedSentences[idx] for idx in testIdx]
        testEncodedSentences = [tensorEncodedSentences[idx] for idx in testIdx]
        testClasses = [classes[idx] for idx in testIdx]
        testDocMapping = [sentenceToDocList[idx] for idx in testIdx]

        if settings["neji"]["use_neji_annotations"] == "True":
            nejiTrainClasses = [nejiClasses[idx] for idx in trainIdx]
            nejiTestClasses = [nejiClasses[idx] for idx in testIdx]
        else:
            nejiTrainClasses = None
            nejiTestClasses = None

        # 100 is the default size used in embedding creation
        max_length = 100
        print("Loaded data successfully.\n")

        modelConfigs = loadModelConfigs(settings)

        DL_model = Model(modelConfigs, BERT_ENTITY_CLASSES, max_length, device)
        print("Model created. Starting training.\n")
        DL_model.train(trainEncodedSentences, trainClasses, neji_classes=nejiTrainClasses)



        print("Starting the testing phase.\n")
        testLabelPred, testLabelTrue = DL_model.test(testEncodedSentences, testClasses, neji_classes=nejiTestClasses)
        print("Finished the testing phase. Evaluating test results\n")
        DL_model.evaluate_test(testLabelPred, testLabelTrue)
   #     # print("Writing model files to disk.\n")
   #     # DL_model.write_model_files(testLabelPred, testLabelTrue, seed)

        print("Generating prediction output for final tsv.\n")
        predFamilyMemberDict, predObservationDict = createOutputTask1(DL_model, testTokenizedSentences,
                                                                      testEncodedSentences, testClasses,
                                                                      testDocMapping, clinicalBERTUtils, neji_classes=nejiTestClasses)
        predFamilyMemberDicts.append(predFamilyMemberDict)
        predObservationsDicts.append(predObservationDict)

    finalFamilyMemberDict = mergeDictionaries(predFamilyMemberDicts)
    finalObservationsDict = mergeDictionaries(predObservationsDicts)

    return finalFamilyMemberDict, finalObservationsDict
