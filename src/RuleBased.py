from rules import *
from stopWords import *
from ProcessorAux import *

PATIENT = "Patient"
PARTNER = "Partner"

#Variables to easly isolate techinques in the pipeline
IDENTIFY_PATIENT = True
COMPLEX_RULES	 = True
EXACT_MATCH		 = True
FIX_DEPENDENTS 	 = True
PREVIOUS_SUBJECT = True
IDENTIFY_PARTNER = False#True
SECTION_RULES 	 = True

class RuleBased(object):
	def processTask1(files):
		result = {}
		for fileName in files:
			fileContent = cleanFile(files[fileName]).split(". ")
			rb = RuleBased()
			rb.process(fileContent)
			result[fileName] = rb.getResults()
		return result, {}#No observations

	def __init__(self):
		self.patientInfo 	= {
			"title" : "",
			"sex" : "",
			"name" : ""
		}
		self.currentSubjects = {
			"fms":[],
			"phrase":"",
			"ls":4
		}
		self.previousSubjects = { #structure equal to currentSubjects
			"fms":[],
			"phrase":""
		}
		self.results 	= list()
		self.isPartner  = False

	def getResults(self):
		return self.results

	def process(self, fileContent):
		lineNr = 0
		for line in fileContent:
			if line.startswith("\n"):
				self.isPartner  = False
			phrase = line.strip()
			result = self.processPhrase(phrase, lineNr)
			if result != None:
				for subject in self.currentSubjects["fms"]:
					if subject["fm"] != PATIENT and subject["fm"] != PARTNER:
						ann = (subject["fm"],self.currentSubjects["phrase"])
						self.results.append(ann)
			self.previousSubjects = self.currentSubjects.copy()
			lineNr += 1

	def processPhrase(self, phrase, lineNr):
		result, markedPhrase, toAddInResults = self.applyRules(phrase, lineNr)
		if result != None:
			if toAddInResults:
				return self.buildResult(result, markedPhrase)
			else:
				return (result, markedPhrase)
		return None

	def applyRules(self, line, lineNr):
		phrase = line
		### Tries to idenfity the patient as subject
		if IDENTIFY_PATIENT:
			if self.isPatientSubject(phrase):
				return self.buildPatientEntry(), phrase, True

		### Tries to idenfity the partner as subject
		if IDENTIFY_PARTNER:
			if self.isPartnerSubject(phrase):
				self.isPartner = True
				return None, phrase, False
			if self.isPartner:#Stop applying rules after identifying the possibility of being partner fms
				return None, phrase, False

		### Tries to identify if its a section
		if SECTION_RULES:
			phrase = self.sectionRules(phrase)

		### Apply the complex rules
		if COMPLEX_RULES:
			results, markedPhrase, toAddInResults = self.complexRulesFirst(phrase)
			if toAddInResults != None:
				return results, markedPhrase, toAddInResults


		### Apply exact match searching
		if EXACT_MATCH:
			results, markedPhrase, fmIndex, readedWords = self.relativeExactMatch(phrase)

			#Fix some matched results based on the previous match (maternal or paternal)
			if PREVIOUS_SUBJECT:
				results = self.fixFamilySideBasedOnPrevious(results)

			if FIX_DEPENDENTS:
				results, markedPhrase, toAddInResults = self.fixDependentRelatives(results, markedPhrase, fmIndex, readedWords)
			else:
				toAddInResults = True
			if toAddInResults != None:
				return results, markedPhrase, toAddInResults

		### to do
		return None, phrase, False


	###############################
	#########    Rules    #########
	###############################
	def sectionRules(self, phrase):
		newPhrase = ""
		for wordInPhrase in phrase.split(" "):
			word = wordInPhrase.replace(",", "")
			if word not in sectionRules:
				newPhrase += " " + word 
		return newPhrase

	def complexRulesFirst(self, phrase):
		entities = set()
		foundNegationRules = False
		count = 0
		readedWords = []
		for wordInPhrase in phrase.split(" "):
			word = wordInPhrase.replace(",", "").replace(":", "")
			if word in STOPWORDS or word in MEDICALSTOPWORDS or numThere(word):
				continue
			for concept, familyMember, sideOfFamily, wb, wa, acceptence in complexTermsFirst:
				if concept == word:
					##Analyse list of words before and after concept (as it is indicated in the list)
					if count >= len(wb) and len(readedWords) - count >= len(wa):
						refBeforeText = readedWords[count-len(wb):count]
						similarBefore = [a for a, b in zip(refBeforeText, wb) if a==b]
						refAfterText = readedWords[count:count+len(wa)]
						similarAfter = [a for a, b in zip(refAfterText, wa) if a==b]
						if len(similarBefore) == len(wb) and len(similarAfter) == len(wa):
							if acceptence:
								entities.add((familyMember, sideOfFamily))
							else:
								foundNegationRules = True
			count += 1
			readedWords += [word]
		if len(entities) != 0:
			results = []
			markedPhrase = phrase
			for fm, sf in entities:
				markedPhrase = getMarkedPhrase(markedPhrase, fm.lower())
				markedPhrase = getMarkedPhrase(markedPhrase, sf.lower())
				results += [{
					"fm": (fm, sf), 
					"sex": self.subjectSex(fm), 
					"ls":-1
				}]
			return results, markedPhrase, True
		if foundNegationRules:
			return None, phrase, False
		return None, None, None

	def relativeExactMatch(self, phrase):
		results = []
		readedWords = []
		count = 0
		block = False
		markedPhrase = phrase
		fmIndex = {}#key: count, value: possession or fm 
		for wordInPhrase in phrase.split(" "):
			word = wordInPhrase.replace(",", "").replace(":", "")
			if word in POSSESSIVEVERBS or word in negationWords or word in SPECIALSTOPWORDS:
				fmIndex[count] = word
				readedWords += [word]
				count += 1
				continue	
			if word in STOPWORDS or word in MEDICALSTOPWORDS or numThere(word):
				continue
			for concept, familyMember, sideOfFamily in simpleTermsSecond:
				if concept == word:
					if count > 0:
						if readedWords[count-1] in negationWords:
							break
					markedPhrase = getMarkedPhrase(markedPhrase, word)
					fm = {
						"fm": (familyMember, sideOfFamily), 
						"sex": self.subjectSex(familyMember), 
						"ls":-1,  
					}
					results += [fm]
					fmIndex[count] = fm 
			readedWords += [word]
			count += 1
		return results, markedPhrase, fmIndex, readedWords

	def fixFamilySideBasedOnPrevious(self, results):
		newResults = []
		for fm in results:
			relative = fm["fm"][0]
			familySide = fm["fm"][1]
			for previousSubjects in self.previousSubjects["fms"]:
				if relative == previousSubjects["fm"][0] and familySide == "NA":
					fm["fm"] = (relative, previousSubjects["fm"][1])
			newResults += [fm]
		return results

	def fixDependentRelatives(self, results, markedPhrase, fmIndex, readedWords):
		if len(results) == 1:
			return results, markedPhrase, True 
		elif len(results) > 1: 
			subjectInList = None
			changeNext = False
			negation = False
			countFM = 0
			filteredResults = []
			for idx in fmIndex:
				if fmIndex[idx] in POSSESSIVEVERBS:
					if subjectInList != None:
						changeNext = True
				elif fmIndex[idx] in negationWords:#Means that relative does not have (no sure)
					negation = True
				elif fmIndex[idx] in SPECIALSTOPWORDS:
					pass#é aqui que removo os half e siblings (or parents or grandparents)
				else:
					if negation:
						negation = False	
					elif changeNext and subjectInList != None:
						base = subjectInList["fm"][0]
						relative = results[countFM]["fm"][0]
						for rel in relations:
							fm1 = rel[0] 
							fm2 = rel[1]
							related = rel[2]
							if base == fm1 and relative == fm2:
								if related != None:
									if len(rel) == 3:
										#results[countFM]["fm"] = (related, subjectInList["fm"][1])#results[countFM]["fm"][1])
										results[countFM]["fm"] = (related, results[countFM]["fm"][1])
									elif len(rel) == 4:
										results[countFM]["fm"] = (related, rel[3])	
									filteredResults += [results[countFM]]
					else:
						subjectInList = fmIndex[idx]
						filteredResults += [results[countFM]]  
					countFM += 1
			return filteredResults, markedPhrase, True
		return None, markedPhrase, False


	###############################
	######### Aux Methods #########
	###############################
	def isPatientSubject(self, phrase):
		lastSubject = self.previousSubjects["fms"]
		if len(lastSubject) == 0 or len(lastSubject) > 1:
			return False
		if self.patientInfo["sex"] == "Female" and lastSubject[0]["fm"] == PATIENT:
			if any([x == phrase.split(" ")[0] for x in femalePatientWordList]):
				return True
		if self.patientInfo["sex"] == "Male" and lastSubject[0]["fm"] == PATIENT:
			if any([x == phrase.split(" ")[0] for x in malePatientWordList]):
				return True
		return False

	def isPartnerSubject(self, phrase):
		for wordInPhrase in phrase.split(" "):
			word = wordInPhrase.replace(",", "").replace(":", "")
			if word in noPatientRelatedWordsList:
				return True
		return False


	def buildPatientEntry(self):
		sex = "" if self.patientInfo["sex"] == "" else self.patientInfo["sex"]
		return {"fm": PATIENT, "sex":sex, "ls":-1}

	def buildResult(self, result, phrase):
		if isinstance(result, list):
			self.currentSubjects = {"fms": result, "phrase": phrase}
		else:
			self.currentSubjects = {"fms": [result], "phrase": phrase}
		return "OK"

	def subjectSex(self, subject):
		if any([x == subject for x in femaleFMSex]):
			return "Female"
		if any([x == subject for x in maleFMSex]):
			return "Male"
		if any([x == subject for x in unisexFMSex]):
			return "Unisex"
		return None