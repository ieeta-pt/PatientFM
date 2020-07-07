import torch
import random
import numpy as np
from sklearn.model_selection import KFold

from Reader import Reader
from NejiAnnotator import readPickle
from Entity import createTrueClasses, createDefaultClasses, ENTITY_CLASSES
from models.utils import classListToTensor, classDictToList, getSentenceList, getSentenceListWithMapping, mergeDictionaries, createOutputTask1, createOutputTask2

from models.Embedding_BiLstmCRF.utils import loadModelConfigs
from models.Embedding_BiLstmCRF.model import Model


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

    embeddingModel = np.load(settings["embeddings"]["biowordvec_original"], allow_pickle=True).item()
    vocab = [key for key in embeddingModel]
    vocabSize = len(vocab)
    embeddingsWeightsMatrix = np.zeros((vocabSize, int(settings["embeddings"]["wordvec_size"])))
    word2Idx = {}
    for i, word in enumerate(vocab):
        word2Idx[word] = i
        embeddingsWeightsMatrix[i] = embeddingModel[word]
    embeddingsWeightsMatrix = torch.tensor(embeddingsWeightsMatrix, dtype=torch.float64).to(device)

    trainTokenizedSentences = getSentenceList(trainTXT, tokenized=True)
    trainEncodedSentences = []
    for sentence in trainTokenizedSentences:
        sentence = [word2Idx[token] for token in sentence]
        trainEncodedSentences.append(torch.tensor(sentence, dtype=torch.long).to(device))

    trainClassesDict = createTrueClasses(trainTXT, trainXML)
    trainClasses = classDictToList(trainClassesDict)
    trainClasses = [classListToTensor(sentenceClasses, datatype=torch.long) for sentenceClasses in trainClasses]

    if settings["neji"]["use_neji_annotations"] == "True":
        nejiTrainClassesDict = readPickle(settings["neji"]["neji_train_pickle_biowordvec"])
        nejiTrainClasses = classDictToList(nejiTrainClassesDict)
        nejiTrainClasses = [classListToTensor(sentenceClasses, datatype=torch.float) for sentenceClasses in nejiTrainClasses]

# 100 is the default size used in embedding creation
    max_length = 100
    print("Loaded data successfully.\n")

    modelConfigs = loadModelConfigs(settings)

    DL_model = Model(modelConfigs, ENTITY_CLASSES, max_length, vocabSize, embeddingsWeightsMatrix, device)
    print("Model created. Starting training.\n")
    DL_model.train(trainEncodedSentences, trainClasses, neji_classes=nejiTrainClasses)

    print("Starting the testing phase.\n")
    reader = Reader(dataSettings=settings, corpus="test")
    testTXT = reader.loadDataSet()

    testClassesDict = createDefaultClasses(testTXT)
    testClasses = classDictToList(testClassesDict)
    testClasses = [classListToTensor(sentenceClasses, datatype=torch.long) for sentenceClasses in testClasses]

    if settings["neji"]["use_neji_annotations"] == "True":
        nejiTestClassesDict = readPickle(settings["neji"]["neji_test_pickle_biowordvec"])
        nejiTestClasses = classDictToList(nejiTestClassesDict)
        nejiTestClasses = [classListToTensor(sentenceClasses, datatype=torch.float) for sentenceClasses in nejiTestClasses]

    testTokenizedSentences, testSentenceToDocList = getSentenceListWithMapping(testTXT, tokenized=True)
    testEncodedSentences = []
    for sentence in testTokenizedSentences:
        sentence = [word2Idx[token] for token in sentence]
        testEncodedSentences.append(torch.tensor(sentence, dtype=torch.long).to(device))

    predFamilyMemberDict, predObservationDict = createOutputTask1(DL_model, testTokenizedSentences, testEncodedSentences,
                                                                  testClasses, testSentenceToDocList, neji_classes=nejiTestClasses)
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
    tokenizedSentences, sentenceToDocList = getSentenceListWithMapping(trainTXT, tokenized=True)

    embeddingModel = np.load(settings["embeddings"]["biowordvec_original"], allow_pickle=True).item()
    vocab = [key for key in embeddingModel]
    vocabSize = len(vocab)
    embeddingsWeightsMatrix = np.zeros((vocabSize, int(settings["embeddings"]["wordvec_size"])))
    word2Idx = {}
    for i, word in enumerate(vocab):
        word2Idx[word] = i
        embeddingsWeightsMatrix[i] = embeddingModel[word]
    embeddingsWeightsMatrix = torch.tensor(embeddingsWeightsMatrix, dtype=torch.float64).to(device)

    encodedSentences = []
    for sentence in tokenizedSentences:
        sentence = [word2Idx[token] for token in sentence]
        encodedSentences.append(torch.tensor(sentence, dtype=torch.long).to(device))

    classesDict = createTrueClasses(trainTXT, trainXML)
    classes = classDictToList(classesDict)
    classes = [classListToTensor(sentenceClasses, datatype=torch.long) for sentenceClasses in classes]

    if settings["neji"]["use_neji_annotations"] == "True":
        nejiClassesDict = readPickle(settings["neji"]["neji_train_pickle_biowordvec"])
        nejiClasses = classDictToList(nejiClassesDict)
        nejiClasses = [classListToTensor(sentenceClasses, datatype=torch.float) for sentenceClasses in nejiClasses]

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

        DL_model = Model(modelConfigs, ENTITY_CLASSES, max_length, vocabSize, embeddingsWeightsMatrix, device)
        print("Model created. Starting training.\n")
        DL_model.train(trainEncodedSentences, trainClasses, neji_classes=nejiTrainClasses)

        # DL_model.train_time_debug(trainTokenizedSentences, trainEmbeddings, trainClasses)

        print("Starting the testing phase.\n")
        testLabelPred, testLabelTrue = DL_model.test(testEncodedSentences, testClasses, neji_classes=nejiTestClasses)
        print("Finished the testing phase. Evaluating test results\n")
        DL_model.evaluate_test(testLabelPred, testLabelTrue)
        print("Writing model files to disk.\n")
        DL_model.write_model_files(testLabelPred, testLabelTrue, seed)

        print("Generating prediction output for final tsv.\n")
        predFamilyMemberDict, predObservationDict = createOutputTask1(DL_model, testTokenizedSentences,
                                                                      testEncodedSentences, testClasses,
                                                                      testDocMapping, neji_classes=nejiTestClasses)
        predFamilyMemberDicts.append(predFamilyMemberDict)
        predObservationsDicts.append(predObservationDict)

    finalFamilyMemberDict = mergeDictionaries(predFamilyMemberDicts)
    finalObservationsDict = mergeDictionaries(predObservationsDicts)

    return finalFamilyMemberDict, finalObservationsDict
