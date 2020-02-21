#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from Reader import Reader
from embeddings.Vocabulary import createVocabularyFile, createFasttextModel
from Preprocessing import nltkSentenceSplit


def runPipeline(settings):

    if not os.path.exists(settings["embeddings"]["vocabulary_path"]):
        sentenceList = list()
        for corpus in ("train", "test"):
            reader = Reader(dataSettings=settings, corpus=corpus)
            filesRead = reader.loadDataSet()

            for fileName in filesRead:
                sentences = nltkSentenceSplit(filesRead[fileName], verbose=False)
                sentenceList.extend(sentence for sentence in sentences)

        createVocabularyFile(sentenceList, settings["embeddings"]["vocabulary_path"], verbose=False)

    if not (os.path.exists(settings["embeddings"]["biowordvec_original"]) or os.path.exists(settings["embeddings"]["biowordvec_normalized"])):
        createFasttextModel(settings["embeddings"]["vocabulary_path"], settings["embeddings"]["wordvec_path"],
                            settings["embeddings"]["biowordvec_original"], settings["embeddings"]["biowordvec_normalized"])




