#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from Reader import Reader
from embeddings.Embeddings import Embeddings, writeEmbeddingsPickle, readEmbeddingsPickle
from embeddings.Vocabulary import createVocabularyFile, createFasttextModel
from Preprocessing import nltkSentenceSplit, nltkTokenize



def runPipeline(settings):

    createVocabulary(settings)
    # createEmbeddingModels(settings)
    for corpus in ("train", "test"):
        createEmbeddingsPickle(settings, corpus)



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


def createEmbeddingModels(settings):
    # Create smaller biowordvec embedding models
    if not (os.path.exists(settings["embeddings"]["biowordvec_original"]) or os.path.exists(settings["embeddings"]["biowordvec_normalized"])):
        createFasttextModel(settings["embeddings"]["vocabulary_path"], settings["embeddings"]["wordvec_path"],
                            settings["embeddings"]["biowordvec_original"], settings["embeddings"]["biowordvec_normalized"])


def createEmbeddingsPickle(settings, corpus):
    if corpus == "train": picklePath = settings["embeddings"]["train_embeddings_pickle"]
    elif corpus == "test": picklePath = settings["embeddings"]["test_embeddings_pickle"]

    if not os.path.exists(picklePath):
        tokenizedSentenceList = list()
        reader = Reader(dataSettings=settings, corpus=corpus)
        filesRead = reader.loadDataSet()
        for fileName in filesRead:
            sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
            for sentence in sentences:
                sentence = nltkTokenize(sentence)
                tokenizedSentenceList.extend([sentence])

        embeddings = Embeddings(settings["embeddings"]["biowordvec_original"], settings["embeddings"]["biowordvec_normalized"],
                                settings["embeddings"]["wordvec_size"])
        embeddingsVec = Embeddings.wordvec_concat(tokenizedSentenceList)
        writeEmbeddingsPickle(embeddingsVec, picklePath)
        print("Created pickle file {}".format(picklePath))



