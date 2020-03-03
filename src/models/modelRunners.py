import torch
import random
import numpy as np
from sklearn.model_selection import KFold

from Reader import Reader
from embeddings.Embeddings import readEmbeddingsPickle
from Entity import createTrueClasses, createDefaultClasses, ENTITY_CLASSES
from models.utils import classListToTensor, createTestOutputTask1, getSentenceList, classDictToList
from models.BiLstmCRF.utils import loadModelConfigs

from models.BiLstmCRF.model import Model


def runModel(settings):
    """
    Trains the model in the FULL training dataset and computes predictions for the FULL test set,
     generating the output for the tsv submission file
    :param settings:
    :return: predFamilyMemberDict, predObservationDict - predictions stored in dictionaries, keyed by filename
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

    reader = Reader(dataSettings=settings, corpus="train")
    filesRead = reader.loadDataSet()
    XMLannotations = reader.loadXMLAnnotations(filesRead)
    trainTokenizedSentences = getSentenceList(filesRead, tokenized=True)
    trainEmbeddings = readEmbeddingsPickle(settings["embeddings"]["train_embeddings_pickle"])
    trainClassesDict = createTrueClasses(filesRead, XMLannotations)
    trainClasses = classDictToList(trainClassesDict)
    trainClasses = [classListToTensor(sentenceClasses) for sentenceClasses in trainClasses]

    # 100 is the default size used in embedding creation
    max_length = 100
    print("Loaded data successfully.\n")

    modelConfigs = loadModelConfigs(settings)
    DL_model = Model(modelConfigs, ENTITY_CLASSES, max_length, device)
    print("Model created. Starting training.\n")
    DL_model.train(trainTokenizedSentences, trainEmbeddings, trainClasses)
    # DL_model.train_time_debug(trainTokenizedSentences, trainEmbeddings, trainClasses)

    print("Starting the testing phase.\n")
    reader = Reader(dataSettings=settings, corpus="test")
    filesRead = reader.loadDataSet()

    predFamilyMemberDict, predObservationDict = createTestOutputTask1(settings, DL_model, filesRead)
    return predFamilyMemberDict, predObservationDict


def runModelDevelopment(settings, trainTXT, trainXML, cvFolds):

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

    tokenizedSentences = getSentenceList(trainTXT, tokenized=True)
    embeddings = readEmbeddingsPickle(settings["embeddings"]["train_embeddings_pickle"])
    classesDict = createTrueClasses(trainTXT, trainXML)
    classes = classDictToList(classesDict)
    classes = [classListToTensor(sentenceClasses) for sentenceClasses in classes]

    kFolds = KFold(n_splits=cvFolds)
    for trainIdx, testIdx in kFolds.split(tokenizedSentences):
        # print(trainIdx)
        # print(testIdx)

        trainTokenizedSentences = [tokenizedSentences[idx] for idx in trainIdx]
        trainEmbeddings = embeddings[trainIdx]
        trainClasses = [classes[idx] for idx in trainIdx]

        testTokenizedSentences = [tokenizedSentences[idx] for idx in testIdx]
        testEmbeddings = embeddings[testIdx]
        testClasses = [classes[idx] for idx in testIdx]

        # 100 is the default size used in embedding creation
        max_length = 100
        print("Loaded data successfully.\n")

        modelConfigs = loadModelConfigs(settings)

        DL_model = Model(modelConfigs, ENTITY_CLASSES, max_length, device)
        print("Model created. Starting training.\n")
        DL_model.train(trainTokenizedSentences, trainEmbeddings, trainClasses)
        # DL_model.train_time_debug(trainTokenizedSentences, trainEmbeddings, trainClasses)
        print("Starting the testing phase.\n")
        testLabelPred, testLabelTrue = DL_model.test(testTokenizedSentences, testEmbeddings, testClasses)
        print("Finished the testing phase. Evaluating test results\n")
        DL_model.evaluate_test(testLabelPred, testLabelTrue)
        print("Writing model files to disk.\n")
        DL_model.write_model_files(testLabelPred, testLabelTrue, seed)


def runModelBothCorpus(settings):
    """
                        CURRENTLY NOT USED
    :param settings:
    :return:
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

    for Corpus in ("train", "test"):
        reader = Reader(dataSettings=settings, corpus=Corpus)
        filesRead = reader.loadDataSet()
        if Corpus == "train":
            XMLannotations = reader.loadXMLAnnotations(filesRead)
            trainTokenizedSentences = getSentenceList(filesRead, tokenized=True)
            trainEmbeddings = readEmbeddingsPickle(settings["embeddings"]["train_embeddings_pickle"])
            trainClassesDict = createTrueClasses(filesRead, XMLannotations)
            trainClasses = classDictToList(trainClassesDict)
            trainClasses = [classListToTensor(sentenceClasses) for sentenceClasses in trainClasses]
        elif Corpus == "test":
            testTokenizedSentences = getSentenceList(filesRead, tokenized=True)
            testEmbeddings = readEmbeddingsPickle(settings["embeddings"]["test_embeddings_pickle"])
            testClassesDict = createDefaultClasses(filesRead)
            testClasses = classDictToList(testClassesDict)
            testClasses = [classListToTensor(sentenceClasses) for sentenceClasses in testClasses]

    # 100 is the default size used in embedding creation
    max_length = 100
    print("Loaded data successfully.\n")

    modelConfigs = loadModelConfigs(settings)

    DL_model = Model(modelConfigs, ENTITY_CLASSES, max_length, device)
    print("Model created. Starting training.\n")
    DL_model.train(trainTokenizedSentences, trainEmbeddings, trainClasses)
    # DL_model.train_time_debug(trainTokenizedSentences, trainEmbeddings, trainClasses)
    print("Starting the testing phase.\n")
    testLabelPred, testLabelTrue = DL_model.test(testTokenizedSentences, testEmbeddings, testClasses)
    print("Finished the testing phase. Evaluating test results\n")
    DL_model.evaluate_test(testLabelPred, testLabelTrue)
    print("Writing model files to disk.\n")
    DL_model.write_model_files(testLabelPred, testLabelTrue, seed)



