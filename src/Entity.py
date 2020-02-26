#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Preprocessing import nltkSentenceSplit, nltkTokenize

## BIO format: B - beginning, I - inside, O - outside entity
## Type of entity: FamilyMember, Observation
## The 13 resulting labels are manually coded in this variable

ENTITY_CLASSES = {"O": 0,
                  "B-Observation": 1,  "I-Observation": 2,
                  "P-FamilyMember": 3, "M-FamilyMember": 4, "NA-FamilyMember": 5}

FAMILY_PLURALS = {'sisters', 'brothers', 'parents', 'mothers', 'fathers',
                   'cousins', 'grandparents', 'grandmothers', 'grandfathers',
                  'sons', 'childs', 'daughters', 'siblings', 'aunts', 'uncles'}

def createDefaultClasses(datasetTXT):
    """
    :param datasetTXT: dict with text from txt files indexed by filename
    :return: Dict with key:filename, value:list of lists with classes per sentence in the document
    """
    classesDict = {}
    for fileName in datasetTXT:
        classesDict[fileName] = []
        sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
        for sentence in sentences:
            sentence = nltkTokenize(sentence)
            classesDict[fileName].append([int(0) for _ in sentence])
    return classesDict

def createTrueClasses(datasetTXT, XMLAnnotations):
    """
    Creates default classes and populates them with the golden standard from the annotations (Observations and FamilyMember annotations only)
    :param datasetTXT:
    :param XMLAnnotations:
    :return: classesDict
    """
    classesDict = createDefaultClasses(datasetTXT)
    classesDict = setObservationClasses(classesDict, datasetTXT, XMLAnnotations)
    classesDict = setFamilyMemberClasses(classesDict, datasetTXT, XMLAnnotations)
    return classesDict

def setObservationClasses(classesDict, datasetTXT, datasetXML):
    """
    This method updates default classes with the classes for Observation entities

    :param classesDict: dict with key:filename, value:list of lists with default classes per sentence in the document
    :param datasetTXT: dict with text from txt files indexed by filename
    :param datasetXML: dict with annotations from xml files indexed by filename
    :return: Dict with key:filename, value:list of lists with classes per sentence in the document
    """
    for fileName in datasetTXT:
        sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
        observations = filterObservations(datasetXML[fileName])
        # Some documents can have no observations. In that case, the default class list remains unchanged
        if observations:
            observationList = [nltkTokenize(observation[0]) for _, observation in observations] #The [0] is to store the string instead of a list with a string
            observationIdx = 0
            for sentencePosition, sentence in enumerate(sentences):
                tokenIdx = 0
                sentence = nltkTokenize(sentence)
                for tokenPosition, token in enumerate(sentence):
                    if observationIdx <= len(observationList) - 1:
                        if token == observationList[observationIdx][tokenIdx]:
                            if tokenIdx == 0:
                                classesDict[fileName][sentencePosition][tokenPosition] = ENTITY_CLASSES["B-Observation"]
                                if len(observationList[observationIdx]) == 1:
                                    observationIdx += 1
                                else:
                                    tokenIdx += 1
                            else:
                                classesDict[fileName][sentencePosition][tokenPosition] = ENTITY_CLASSES["I-Observation"]
                                tokenIdx += 1
                                if tokenIdx == len(observationList[observationIdx]):
                                    tokenIdx = 0
                                    observationIdx += 1
    return classesDict

def setFamilyMemberClasses(classesDict, datasetTXT, datasetXML):
    """
    This method updates default classes with the classes for FamilyMember entities

    :param classesDict: dict with key:filename, value:list of lists with default classes per sentence in the document
    :param datasetTXT: dict with text from txt files indexed by filename
    :param datasetXML: dict with annotations from xml files indexed by filename
    :return: Dict with key:filename, value:list of lists with classes per sentence in the document
    """
    for fileName in datasetTXT:
        sentences = nltkSentenceSplit(datasetTXT[fileName], verbose=False)
        familyMembers = filterFamilyMembers(datasetXML[fileName])
        # Some documents can have no family member mentions. In that case, the class list remains unchanged
        if familyMembers:
            familyMemberList = [(familyMember.lower(), familySide) for _, familyMember, familySide in familyMembers]
            familyMemberIdx = 0
            for sentencePosition, sentence in enumerate(sentences):
                sentence = nltkTokenize(sentence)
                for tokenPosition, token in enumerate(sentence):
                    if familyMemberIdx <= len(familyMemberList) - 1:
                        # Some tokens can be in the plural but the annotation is singular, we check that with the condition after "or"
                        if token == familyMemberList[familyMemberIdx][0] or (token in FAMILY_PLURALS and token[:-1] == familyMemberList[familyMemberIdx][0]):
                            if familyMemberList[familyMemberIdx][1] == "Paternal":
                                classesDict[fileName][sentencePosition][tokenPosition] = ENTITY_CLASSES["P-FamilyMember"]
                            elif familyMemberList[familyMemberIdx][1] == "Maternal":
                                classesDict[fileName][sentencePosition][tokenPosition] = ENTITY_CLASSES["M-FamilyMember"]
                            elif familyMemberList[familyMemberIdx][1] == "NA":
                                classesDict[fileName][sentencePosition][tokenPosition] = ENTITY_CLASSES["NA-FamilyMember"]
                            familyMemberIdx += 1
    return classesDict

def filterObservations(annotationDict):
    filteredList = [(annotation['spans'], annotation['mentions']) for _, annotation in annotationDict.items() if annotation['type'] == 'Observation']
    filteredList = sorted(filteredList)
    return filteredList

def filterFamilyMembers(annotationDict):
    filteredList = [(annotation['spans'], annotation['familyRelation'], annotation['familySide']) for _, annotation in annotationDict.items() if annotation['type'] == 'FamilyMember']
    filteredList = sorted(filteredList)
    return filteredList