import torch
import random
import numpy as np
from sklearn.model_selection import KFold

from Reader import Reader
from embeddings.Embeddings import readEmbeddingsPickle
from Entity import createTrueClasses, createDefaultClasses, ENTITY_CLASSES
from models.utils import classListToTensor, createTestOutputTask1, createTrainOutputTask1, classDictToList, getSentenceList, getSentenceListWithMapping, mergeDictionaries

from models.Embedding_BiLstmCRF.utils import loadModelConfigs
from models.Embedding_BiLstmCRF.model import Model


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

    print("Loading and preprocessing data.\n")
    tokenizedSentences, sentenceToDocList = getSentenceListWithMapping(trainTXT, tokenized=True)

    embeddingModel = np.load(settings["embeddings"]["biowordvec_original"], allow_pickle=True).item()
    vocab = [key for key in embeddingModel]
    vocabSize = len(vocab)
    embeddingsWeightsMatrix = np.zeros((vocabSize, int(settings["embeddings"]["wordvec_size"])))
    word2Idx = {}
    for i, word in enumerate(vocab):
        word2Idx[word] = i
        embeddingsWeightsMatrix[i] = embeddingModel[word]
    embeddingsWeightsMatrix = torch.tensor(embeddingsWeightsMatrix, dtype=torch.float64)

    encodedSentences = []
    for sentence in tokenizedSentences:
        sentence = [word2Idx[token] for token in sentence]
        encodedSentences.append(torch.tensor(sentence, dtype=torch.long))

    classesDict = createTrueClasses(trainTXT, trainXML)
    classes = classDictToList(classesDict)
    classes = [classListToTensor(sentenceClasses) for sentenceClasses in classes]

    kFolds = KFold(n_splits=cvFolds)
    predFamilyMemberDicts = []
    predObservationsDicts = []

    print("Beginning KFold cross validation.\n")

    for trainIdx, testIdx in kFolds.split(tokenizedSentences):
        trainEncodedSentences = [encodedSentences[idx] for idx in trainIdx]
        trainClasses = [classes[idx] for idx in trainIdx]

        testTokenizedSentences = [tokenizedSentences[idx] for idx in testIdx]
        testEncodedSentences = [encodedSentences[idx] for idx in testIdx]
        testClasses = [classes[idx] for idx in testIdx]
        testDocMapping = [sentenceToDocList[idx] for idx in testIdx]

        # 100 is the default size used in embedding creation
        max_length = 100
        print("Loaded data successfully.\n")

        modelConfigs = loadModelConfigs(settings)

        DL_model = Model(modelConfigs, ENTITY_CLASSES, max_length, vocabSize, embeddingsWeightsMatrix, device)
        print("Model created. Starting training.\n")
        DL_model.train(trainEncodedSentences, trainClasses)

        # DL_model.train_time_debug(trainTokenizedSentences, trainEmbeddings, trainClasses)

        print("Starting the testing phase.\n")
        testLabelPred, testLabelTrue = DL_model.test(testEncodedSentences, testClasses)
        print("Finished the testing phase. Evaluating test results\n")
        DL_model.evaluate_test(testLabelPred, testLabelTrue)
        print("Writing model files to disk.\n")
        DL_model.write_model_files(testLabelPred, testLabelTrue, seed)


        print("Generating prediction output for final tsv.\n")
        predFamilyMemberDict, predObservationDict = createTrainOutputTask1(DL_model, testTokenizedSentences, testEmbeddings, testClasses, testDocMapping)
        predFamilyMemberDicts.append(predFamilyMemberDict)
        predObservationsDicts.append(predObservationDict)

    finalFamilyMemberDict = mergeDictionaries(predFamilyMemberDicts)
    finalObservationsDict = mergeDictionaries(predObservationsDicts)

    return finalFamilyMemberDict, finalObservationsDict





