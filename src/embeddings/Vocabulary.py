#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from Preprocessing import nltkTokenize

def createVocabularyFile(sentenceList, vocabularyFilePath, verbose=False):
    t0 = time.time()
    print('Building vocabulary... \n', end='')
    datasetVocab = set()
    datasetVocab.update(token for sentence in sentenceList for token in nltkTokenize(sentence))
    t1 = time.time()
    if verbose:
        print(' ({} seconds)'.format(t1 - t0))
        print('Sorting vocabulary...', end='')
    datasetVocab = sorted(datasetVocab)
    t2 = time.time()
    if verbose:
        print(' ({} seconds)'.format(t2 - t1))
        print('Length of datasetVocab): {}'.format(len(datasetVocab)))
        print('datasetVocab[:5]: {}'.format(datasetVocab[:5]))

    with open(vocabularyFilePath, mode='w', encoding='utf-8') as file:
        file.write('\n'.join(datasetVocab) + '\n')
    print("Vocabulary.txt created.")


def createFasttextModel(vocabularyFilePath, biowordvecPath, biowordvecVocabOrigPath, biowordvecVocabNormPath):
    from gensim.models import FastText
    import numpy as np
    from sklearn.preprocessing import normalize
    from time import time

    fpath = biowordvecPath

    t0 = time()
    model = FastText.load_fasttext_format(fpath)
    t1 = time()
    print(' ({} seconds)'.format(t1 - t0))

    # Vocabulary.
    vocab = model.wv.vocab

    # 16545452 unique tokens.
    n = len(vocab)

    f = open(vocabularyFilePath, mode='r', encoding='utf-8')
    vocab = f.read().splitlines()
    f.close()

    biowordvecVocabOrig = dict()
    biowordvecVocabNorm = dict()

    for word in vocab:
        vec = model[word]
        biowordvecVocabOrig[word] = vec
        biowordvecVocabNorm[word] = np.float32(normalize([vec])[0])

    np.save(biowordvecVocabOrigPath, biowordvecVocabOrig)
    np.save(biowordvecVocabNormPath, biowordvecVocabNorm)

    # To load:
    # biowordvec_vocab_orig = np.load(biowordvecVocabOrigPath, allow_pickle=True).item()
    # biowordvec_vocab_norm = np.load(biowordvecVocabNormPath, allow_pickle=True).item()

