import pickle
import numpy as np
from sklearn.preprocessing import normalize


class Embeddings:

    def __init__(self, wordvecVocabOrig, wordvecVocabNorm, WORDVEC_SIZE):
        self.WV = np.load(wordvecVocabOrig, allow_pickle=True).item()
        self.WV_NORM = np.load(wordvecVocabNorm, allow_pickle=True).item()
        self.WORDVEC_SIZE = WORDVEC_SIZE

    def wordvec_concat(self, tokenized_sentences, maxlen=100, pre_norm=False,
                       post_norm=False):
        n = len(tokenized_sentences)
        assert n >= 1
        assert maxlen >= 1
        model = self.WV_NORM if pre_norm else self.WV
        vecs = np.zeros((n, maxlen * self.WORDVEC_SIZE), dtype='float32')
        # print(vecs.shape)
        for i, s in enumerate(tokenized_sentences):
            for j, w in enumerate(s[:maxlen]):
                vecs[i][j*self.WORDVEC_SIZE:(j+1)*self.WORDVEC_SIZE] = model[w]
        return normalize(vecs) if post_norm else vecs


def writeEmbeddingsPickle(embeddingsVec, picklePath):
    with open(picklePath, 'wb') as pickle_handle:
        pickle.dump(embeddingsVec, pickle_handle, protocol=4)


def readEmbeddingsPickle(picklePath):
    embeddings = []
    with open(picklePath, 'rb') as pickle_handle:
        embeddings = pickle.load(pickle_handle)
    return embeddings