#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from Reader import Reader
from Entity import createTrueClasses, createDefaultClasses, ENTITY_CLASSES
from Preprocessing import nltkSentenceSplit, nltkTokenize
from embeddings.Embeddings import Embeddings, writeEmbeddingsPickle, readEmbeddingsPickle
from embeddings.Vocabulary import createVocabularyFile, createFasttextModel
from models.BiLstmCRF.utils import classListToTensor
from models.BiLstmCRF.model import Model, loadModelConfigs


def runEmbeddingCreationPipeline(settings):

    createVocabulary(settings)
    createEmbeddingModels(settings)
    for corpus in ("train", "test"):
        createEmbeddingsPickle(settings, corpus)

def runModelDevelopment(settings, trainTXT, trainXML):

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

    trainPercentage = 0.70
    trainTestSplit = round(trainPercentage * (len(tokenizedSentences)))

    trainTokenizedSentences = tokenizedSentences[:trainTestSplit]
    trainEmbeddings = embeddings[:trainTestSplit]
    trainClasses = classes[:trainTestSplit]

    testTokenizedSentences = tokenizedSentences[trainTestSplit:]
    testEmbeddings = embeddings[trainTestSplit:]
    testClasses = classes[trainTestSplit:]

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


def createVocabulary(settings):
    if not os.path.exists(settings["embeddings"]["vocabulary_path"]):
        sentenceList = list()
        for corpus in ("train", "test"):
            reader = Reader(dataSettings=settings, corpus=corpus)
            filesRead = reader.loadDataSet()
            for fileName in filesRead:
                sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
                sentenceList.extend(sentence for sentence in sentences)
        createVocabularyFile(sentenceList, settings["embeddings"]["vocabulary_path"], verbose=False)

def getSentenceList(filesRead, tokenized=False):
    sentenceList = list()
    for fileName in filesRead:
        sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
        if tokenized:
            sentenceList.extend(nltkTokenize(sentence) for sentence in sentences)
        else:
            sentenceList.extend(sentence for sentence in sentences)
    return sentenceList

def classDictToList(classDict):
    classList = list()
    for fileName in classDict:
        classList.extend(sentenceClasses for sentenceClasses in classDict[fileName])
    return classList

def createEmbeddingModels(settings):
    # Create smaller biowordvec embedding models
    if not (os.path.exists(settings["embeddings"]["biowordvec_original"]) or os.path.exists(settings["embeddings"]["biowordvec_normalized"])):
        print("just testing")
        createFasttextModel(settings["embeddings"]["vocabulary_path"], settings["embeddings"]["wordvec_path"],
                            settings["embeddings"]["biowordvec_original"], settings["embeddings"]["biowordvec_normalized"])


def createEmbeddingsPickle(settings, corpus):
    if corpus == "train": picklePath = settings["embeddings"]["train_embeddings_pickle"]
    elif corpus == "test": picklePath = settings["embeddings"]["test_embeddings_pickle"]

    tokenizedSentenceList = list()
    if not os.path.exists(picklePath):
        reader = Reader(dataSettings=settings, corpus=corpus)
        filesRead = reader.loadDataSet()
        for fileName in filesRead:
            sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
            for sentence in sentences:
                sentence = nltkTokenize(sentence)
                tokenizedSentenceList.extend([sentence])

        embeddings = Embeddings(settings["embeddings"]["biowordvec_original"], settings["embeddings"]["biowordvec_normalized"],
                                int(settings["embeddings"]["wordvec_size"]))

        embeddingsVec = embeddings.wordvec_concat(tokenizedSentenceList)
        writeEmbeddingsPickle(embeddingsVec, picklePath)
        print("Created pickle file {}".format(picklePath))

