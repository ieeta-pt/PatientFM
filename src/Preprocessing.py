import os
import nltk

def nltkInitialize(nltk_dir):
    if not os.path.exists(nltk_dir):
        nltk.download('punkt', download_dir=nltk_dir)
        print("NLTK sources downloaded to: {}".format(nltk_dir))
    nltk.data.path.append(nltk_dir)
    print("NLTK sources loaded.")

def nltkSentenceSplit(sentences, verbose=False):
    sentences = nltk.sent_tokenize(sentences)
    if verbose:
        print(sentences)
    return sentences

def nltkTokenize(sentence, verbose=False):
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    if verbose:
        print(sentence)
    return sentence

