# Code for class NejiAnnotator adapted from Sérgio Matos

import os
import re
import json
import pickle
import requests
from Reader import Reader
from Preprocessing import nltkSentenceSplit, nltkTokenize


class NejiAnnotator():
    def __init__(self, ignore=None):
        self.url = "https://bioinformatics.ua.pt/nejiws/annotate/Disorders/annotate"
        self.headers = {'Content-Type': 'application/json; charset=UTF-8'}
        self.sem_types = set(['T020', 'T190', 'T049', 'T019', 'T047', 'T050', 'T037', 'T048', 'T191', 'T046', 'T184', 'T033'])
        self.ignore = ignore

        self.observationsRE = re.compile('[^a-z](%s)[^a-z]' %('|'.join(sorted([line.split('\t')[0] for line in open('../dataset/extra_data/observation_suffixes').read().splitlines() if not line.split('\t')[0].startswith('#') and int(line.split('\t')[1])>=30], key=lambda t: len(t), reverse=True))))
        self.semevalRE = re.compile('[^a-z](%s)[^a-z]' %('|'.join(sorted([line.split('\t')[0] for line in open('../dataset/extra_data/semeval_tps.txt').read().splitlines() if not line.split('\t')[0].startswith('#') and len(line.split('\t')[0])>2 and int(line.split('\t')[1])>=10], key=lambda t: len(t), reverse=True))))
        self.historyRE = re.compile('family medical history|family history|accident')

    def annotate(self, text):
        payload = json.dumps({"text": "%s" %text.lower()}, ensure_ascii=True).encode('utf-8')
        response = requests.request("POST", self.url, data=payload, headers=self.headers)
        try:
            results = json.loads(response.text)
            results = results['entities']

            for match in self.observationsRE.finditer(text.lower()):
                results.append('%s|UMLS:C0000000:T047:Disease|%i' %(match.group(1), match.start(1)))

            for match in self.semevalRE.finditer(text.lower()):
                results.append('%s|UMLS:C0000000:T047:Disease|%i' %(match.group(1), match.start(1)))

            results.sort(key=lambda e: int(e.split('|')[-1]))
            #if docid=='doc_150': print('\n'.join(results))

            entities = []

            # clean-up annotations
            last_entity_index = -1
            for i1, entity1 in enumerate(results):
                if i1 <= last_entity_index: continue

                last_entity_index = i1
                text_mention1, entity_id1, start_index1 = entity1.split('|')
                start_index1 = int(start_index1)
                end_index1 = start_index1 + len(text_mention1) - 1

                #if text_mention1.lower() in self.ignore: continue

                for i2 in range(i1+1, len(results)):
                    entity2 = results[i2]
                    text_mention2, entity_id2, start_index2 = entity2.split('|')
                    start_index2 = int(start_index2)
                    end_index2 = start_index2 + len(text_mention2) - 1

                    #if text_mention2.lower() in self.ignore: continue

                    # merge entities
                    if start_index2 == end_index1 + 2:
                        text_mention1 = text_mention1 + " " + text_mention2

                    elif start_index2 > start_index1 and start_index2 < end_index1 + 2:
                        if end_index2 > end_index1:
                            text_mention1 = text_mention1 + text_mention2[end_index1-start_index2+1:]

                    else:
                        break

                    end_index1 = end_index2
                    entity_id1 = entity_id1 + ";" + entity_id2
                    last_entity_index = i2

                entity1 = '%s|%s|%i' %(text_mention1, entity_id1, start_index1)
                semtype_list = [eid.split(':')[2] for eid in entity_id1.split(';') if eid.split(':')[2] in self.sem_types]
                if semtype_list and not text_mention1.lower() in self.ignore and not self.historyRE.search(text_mention1.lower()):
                    entities.append(entity1)

            #if docid=='doc_150': print('\n'.join(entities))
            return entities

        except Exception as e:
            print(e)
            return None

def createIgnoreSet():
    """
    Creates ignore set for Neji Annotator
    :return: ignore: set with entries to ignore during annotation with Neji
    """

    gs_observations = set()
    ignore = set()
    with open('../dataset/Train/train_subtask1_2.tsv') as fin:
        for line in fin:
            data = line.strip('\n').split('\t')
            if data[1] == 'Observation':
                gs_observations.add(data[2].lower().strip())

    # ignore = set(open('train_FPs.txt').read().splitlines())
    ignore |= set([line.split('\t')[0].lower() for line in open('../dataset/extra_data/semeval_fps.txt').read().splitlines() if int(line.split('\t')[1])>=10])
    stopwords = set(open('../dataset/extra_data/stopwords').read().splitlines())

    gs_observations_tokens = set()
    with open('../dataset/Train/train_subtask1_2.tsv') as fin:
        for line in fin:
            data = line.strip('\n').split('\t')
            if data[1] == 'Observation':
                gs_observations_tokens.update([t for t in data[2].lower().split() if t not in stopwords])

    for line in open('../dataset/extra_data/semeval_fps.txt').read().splitlines():
        term = line.split('\t')[0]
        if not gs_observations_tokens.intersection(set([t for t in term.split() if t not in stopwords])):
            ignore.add(term)

    ignore.discard(gs_observations)
    return ignore

# def createNejiClasses(datasetTXT):
#     """
#     Create "classes" for entities annotated by Neji Annotator
#     :param datasetTXT:
#     :return: classesDict
#     """
#     classesDict = {}
#     NEJIclass = 1
#     NEJIclass = {"B-Annotation": 1, "I-Annotation": 2}
#
#     ignore = createIgnoreSet()
#     neji = NejiAnnotator(ignore=ignore)
#
#     for fileName in datasetTXT:
#         print("Going in file {}".format(fileName))
#         classesDict[fileName] = []
#         sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
#         for sentence in sentences:
#             entities = neji.annotate(sentence)
#             sentence = nltkTokenize(sentence)
#             classList = [int(0) for _ in sentence]
#             if entities:
#                 entities = [entity.split('|')[0] for entity in entities]
#                 entities = unique(entities)
#                 entities = [nltkTokenize(entity) for entity in entities]
#                 entityIdx = 0
#                 maxEntityIdx = len(entities) - 1
#                 tokenIdx = 0
#                 for tokenPosition, token in enumerate(sentence):
#                     if entityIdx <= maxEntityIdx:
#                         if token == entities[entityIdx][tokenIdx]:
#                             classList[tokenPosition] = NEJIclass
#                             if tokenIdx < len(entities[entityIdx]) - 1:
#                                 tokenIdx += 1
#                             else:
#                                 tokenIdx = 0
#                                 entityIdx += 1
#                     else:
#                         break
#
#             classesDict[fileName].append(classList)
#     return classesDict

def createNejiClasses(datasetTXT):
    """
    Create "classes" for entities annotated by Neji Annotator
    :param datasetTXT:
    :return: classesDict
    """
    classesDict = {}
    NEJIclass = {"B-Annotation": 1, "I-Annotation": 2}

    ignore = createIgnoreSet()
    neji = NejiAnnotator(ignore=ignore)

    for fileName in datasetTXT:
        print("Going in file {}".format(fileName))
        classesDict[fileName] = []
        sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
        for sentence in sentences:
            entities = neji.annotate(sentence)
            sentence = nltkTokenize(sentence)
            classList = [int(0) for _ in sentence]
            if entities:
                entities = [entity.split('|')[0] for entity in entities]
                entities = unique(entities)
                entities = [nltkTokenize(entity) for entity in entities]
                entityIdx = 0
                maxEntityIdx = len(entities) - 1
                tokenIdx = 0
                for tokenPosition, token in enumerate(sentence):
                    if entityIdx <= maxEntityIdx:
                        if token == entities[entityIdx][tokenIdx]:
                            if tokenIdx == 0:
                                classList[tokenPosition] = NEJIclass["B-Annotation"]
                            elif tokenIdx > 0:
                                classList[tokenPosition] = NEJIclass["I-Annotation"]

                            if tokenIdx < len(entities[entityIdx]) - 1:
                                tokenIdx += 1
                            else:
                                tokenIdx = 0
                                entityIdx += 1
                    else:
                        break

            classesDict[fileName].append(classList)
    return classesDict

def createNejiSourcesGeneric(datasetTXT, picklePath):
    """
    Creates a pickle file with neji annotations given any input data (in txt format)
    :param datasetTXT:
    :param picklePath:
    :return:
    """
    classesDict = createNejiClasses(datasetTXT)
    writePickle(classesDict, picklePath)

def createNejiSourcesFromCorpus(settings, corpus, picklePath):
    """
    Creates pickle file with class annotations for the given corpus
    :param settings:
    :param corpus:
    :param picklePath:
    :return:
    """
    reader = Reader(dataSettings=settings, corpus=corpus)
    filesRead = reader.loadDataSet()
    classesDict = createNejiClasses(filesRead)
    writePickle(classesDict, picklePath)

def runNejiSourcesCreation(settings):
    """
    If neji class annotations do not exist, this pipeline creates them for the train and test corpus
    :param settings:
    :return:
    """
    for corpus in ("train", "test"):
        if corpus == "train": picklePath = settings["neji"]["neji_train_pickle"]
        elif corpus == "test": picklePath = settings["neji"]["neji_test_pickle"]

        if not os.path.exists(picklePath):
            print("Creating neji annotations for the {} set.".format(corpus))
            reader = Reader(dataSettings=settings, corpus=corpus)
            filesRead = reader.loadDataSet()
            classesDict = createNejiClasses(filesRead)
            writePickle(classesDict, picklePath)
        else:
            print("Pickle file for {} set already exists at {}".format(corpus, picklePath))


def unique(list):
    """
    Creates a list with unique entries, sorted by their appearance in the list (to make an ordered set)
    :param list:
    :return:
    """
    existing = set()
    return [x for x in list if not (x in existing or existing.add(x))]

def writePickle(classDict, picklePath):
    """
    :param classDict: dict indexed by filename with classes for each sentence
    :param picklePath: path for the file to be written
    :return:
    """
    with open(picklePath, 'wb') as pickle_handle:
        pickle.dump(classDict, pickle_handle, protocol=4)

def readPickle(picklePath):
    """
    :param picklePath: path for the file to be read
    :return:
    """
    classDict = []
    with open(picklePath, 'rb') as pickle_handle:
        classDict = pickle.load(pickle_handle)
    return classDict
