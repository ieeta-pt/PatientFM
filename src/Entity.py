#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Preprocessing import nltkSentenceSplit, nltkTokenize

## BIO format: B - beginning, I - inside, O - outside entity
## Type of entity: FamilyMember, Observation
## The 13 resulting labels are manually coded in this variable

ENTITY_CLASSES = {"O": 0,
                  "B-Observation": 1,  "I-Observation": 2,
                  "P-FamilyMember": 3, "M-FamilyMember": 4, "NA-FamilyMember": 5}

def createClasses(datasetTXT, datasetXML):
    """
    :param datasetTXT: dict with text from txt files indexed by filename
    :param datasetXML: dict with annotations from xml files indexed by filename
    :return: Dict with key:filename, value:list of lists with classes per sentence in the document
    """
    classesDict = {}
    for fileName in datasetTXT:
        classesDict[fileName] = []
        sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
        observations = filterObservations(datasetXML[fileName])
        print(fileName)
        # Some documents can have no observations
        if not observations:
            for sentence in sentences:
                sentence = nltkTokenize(sentence)
                classesDict[fileName].append([int(0) for _ in sentence])

        else:
            observationList = [nltkTokenize(observation[0]) for _, observation in observations if observations]
            observationIdx = 0
            for sentence in sentences:
                classList = []
                tokenIdx = 0
                sentence = nltkTokenize(sentence)
                for token in sentence:
                    if observationIdx <= len(observationList) - 1:
                        if token == observationList[observationIdx][tokenIdx]:
                            if tokenIdx == 0:
                                classList.append(ENTITY_CLASSES["B-Observation"])
                                if len(observationList[observationIdx]) == 1:
                                    observationIdx += 1
                                else:
                                    tokenIdx += 1
                            else:
                                classList.append(ENTITY_CLASSES["I-Observation"])
                                tokenIdx += 1
                                if tokenIdx == len(observationList[observationIdx]):
                                    tokenIdx = 0
                                    observationIdx += 1
                        else:
                            classList.append(ENTITY_CLASSES["O"])
                    else:
                        classList.append(ENTITY_CLASSES["O"])
                #assert values are ints
                classList = [int(i) for i in classList]

                print(classList)
                classesDict[fileName].append(classList)
    return classesDict


def filterObservations(annotationDict):
    filteredList = [(annotation['spans'], annotation['mentions']) for _, annotation in annotationDict.items() if annotation['type'] == 'Observation']
    filteredList = sorted(filteredList)
    return filteredList