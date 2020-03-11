from stopWords import *
from rules import *

def numThere(value):
	s = value
	for x in [",", "s", "'s", "-year-old", "years", "-month-old", "-days-old"]:
		s = s.replace(x, "")
	return all(i.isdigit() for i in s)
	
def getMarkedPhrase(phrase, value):
	return phrase#.replace(value+ " ", WARNING + value+ " " + ENDC)

def cleanFile(inFile):
	file = inFile.replace("\n", " ").replace("’", "\'")
	for ch in ['\\','\"','*','_','{','}','[',']','(',')','>','#','+',',','!','$',':',';']:
			if ch in file:
				file = file.replace(ch,"")
	file = file.lower()
	file = file.replace("dr.", "dr").replace("mr.", "mr").replace("mrs.", "mrs").replace("ms.", "ms")
	return file

def fulfillPatientShits(phrase, patientInfo):
	nextIsTheName = False
	dirtyPreviousWord = ""
	for word in phrase.split(" "):
		if word in STOPWORDS:
			dirtyPreviousWord = word.strip()
			if nextIsTheName:
				nextIsTheName = False
			continue
		if nextIsTheName:
			nextIsTheName = False
			if patientInfo["name"] == "":
				if word.endswith("'s"):
					patientInfo["name"] = word[:-2]
				else:
					patientInfo["name"] = word
		if any([x == word for x in prenoms]) and patientInfo["title"] == "" and \
			dirtyPreviousWord != "by":
			patientInfo["title"] = word
			if patientInfo["sex"] == "": 
				if word == "mr":
					patientInfo["sex"] = "Male"
				else:
					patientInfo["sex"] = "Female"
			nextIsTheName = True
		#Phrase maybe related with the patient
		if any([x == word for x in patientWordList]):
			maybePatient = True
			for fm in familyMembersInverted + familyMembers:
				if fm.lower() in phrase:
					maybePatient = False
			if maybePatient:
				for word2 in phrase.split(" "):
					wordSecondary = word2.replace(",", "")
					if patientInfo["sex"] == "" and any([x == wordSecondary for x in femalePatientWordList]):
						patientInfo["sex"] = "Female"
					if patientInfo["sex"] == "" and any([x == wordSecondary for x in malePatientWordList]):
						patientInfo["sex"] = "Male"
		dirtyPreviousWord = word.strip()
	return patientInfo