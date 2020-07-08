import glob
import codecs
#from stopWords import STOPWORDS, SPECIALSTOPWORDS

import xml.etree.ElementTree as ET


class Reader(object):
	def __init__(self, dataSettings, corpus):
		if corpus == "train": self.dataSetDir = dataSettings["datasets"]["train_files"]
		elif corpus == "test": self.dataSetDir = dataSettings["datasets"]["test_files"]
		self.vocFiles = dataSettings["vocabulary"]

	def loadDataSet(self, cleaning=False):
		filesContent = {}
		allFiles = sorted(glob.glob('{}*.{}'.format(self.dataSetDir, "txt")))
		for file in allFiles:
			fileName = file.split("/")[-1].split(".")[0]
			with codecs.open(file, 'r', encoding='utf8') as fp:
				if cleaning:
					filesContent[fileName] = self.cleanFile(fp.read())
				else:
					filesContent[fileName] = fp.read()
		return filesContent

	def cleanFile(self, inFile):
		file = inFile.replace("\n", " ").replace("’", "\'")
		for ch in ['\\','\"','*','_','{','}','[',']','(',')','>','#','+',',','!','$',':',';']:
				if ch in file:
					file = file.replace(ch,"")
		file = file.lower()
		file = file.replace("dr.", "dr").replace("mr.", "mr").replace("mrs.", "mrs").replace("ms.", "ms")
		return file

	def loadXMLAnnotations(self, dataset):
		"""
		Loads XML file tree and extracts information from annotations (entities and relations).
		params: dataset - dict with text from txt files indexed by filename

		returns: xmlFilesContent - Dict of dicts with base dict indexed by file name, "sub"dict indexed by id
			eg. xmlFilesContent[fileName] = {id: {'span': span, 'type': annotationType, "familyRelation": familyRelation, "count": count, "familySide": familySide}})
		"""

		xmlFilesContent = {}
		allFiles = sorted(glob.glob('{}*.{}'.format(self.dataSetDir, "xml")))
		for file in allFiles:
			fileName = file.split("/")[-1].split(".")[0]
			xmlFilesContent[fileName] = {}
			fileTree = ET.parse(file)
			xmlFilesContent = self.extractXMLEntity(fileTree, fileName, xmlFilesContent, dataset[fileName])
			xmlFilesContent = self.extractXMLRelation(fileTree, fileName, xmlFilesContent)
		return xmlFilesContent

	def extractXMLEntity(self, fileElementTree, fileName, xmlFilesContent, txtFileRead):
		"""
		Searches the ElementTree for annotations regarding Entities and updates the filesContent dict with extracted information
		:return: xmlFilesContent
		"""
		for entity in fileElementTree.iter('entity'):
			id             = entity.find('id').text
			span           = entity.find('span').text
			annotationType = entity.find('type').text
			properties 	   = entity.find('properties')

			numSpans, spanTuple = self.spanToTuple(span)

			if annotationType == "Age":
				ageType = properties.find('AgeType').text
				xmlFilesContent[fileName].update({id: {'spans': spanTuple, 'numSpans': numSpans, 'type': annotationType, "ageType": ageType}})

			elif annotationType == "FamilyMember":
				count 		   = properties.find('Count').text
				familyRelation = properties.find('Relation').text
				familySide     = properties.find('SideOfFamily').text
				mentions 	   = self.fetchMentionFromSpan(spanTuple, txtFileRead)
				xmlFilesContent[fileName].update({id: { 'spans': spanTuple, 
														'numSpans': numSpans, 
														'type': annotationType, 
														"familyRelation": familyRelation, 
														"count": count, 
														"familySide": familySide,
														"mention": mentions}})

			elif annotationType == "Observation":
				negation  = properties.find('Negation').text
				certainty = properties.find('Certainty').text
				mentions = self.fetchMentionFromSpan(spanTuple, txtFileRead)
				xmlFilesContent[fileName].update({id: {'spans': spanTuple, 'numSpans': numSpans, 'type': annotationType, 'mentions': mentions, "negation": negation, "certainty": certainty}})

			elif annotationType == "LivingStatus":
				alive   = properties.find('Alive').text
				healthy = properties.find('Healthy').text
				xmlFilesContent[fileName].update({id: {'spans': spanTuple, 'numSpans': numSpans, 'type': annotationType, "alive": alive, "healthy": healthy}})
		return xmlFilesContent

	def extractXMLRelation(self, fileElementTree, fileName, xmlFilesContent):
		"""
		Searches the ElementTree for annotations regarding Relations and updates the filesContent dict with extracted information
		:return: xmlFilesContent
		"""

		for relation in fileElementTree.iter('relation'):
			id             = relation.find('id').text
			annotationType = relation.find('type').text
			properties 	   = relation.find('properties')
			familyMembers  = properties.find('FamilyMembers').text
			sub_properties = [sub_property.text for sub_property in properties.findall('Properties')]
			xmlFilesContent[fileName].update({id: {'type': annotationType, "familyMembers": familyMembers, "Properties": sub_properties}})
		return xmlFilesContent

	def spanToTuple(self, span):
		"""
		Converts a span in string format to a list with span tuples.
		Addresses discontiguous spans (e.g. "128,132;145,150") by creating a tuple per sub-span

		:param: span (in string)
		:return: list of tuples with beginning and end of spans
		"""
		try:
			spanBegin, spanEnd = span.split(',')
		except ValueError:
			numSpans = len(span.split(';'))
			spans = [subSpan.split(',') for subSpan in span.split(';')]
			spanTuple = [(int(spanBegin), int(spanEnd)) for spanBegin, spanEnd in spans]
		else:
			numSpans = 1
			spanTuple = [(int(spanBegin), int(spanEnd))]
		return numSpans, spanTuple

	def fetchMentionFromSpan(self, spanTuple, txtFileRead):
		observations = []
		for span in spanTuple:
			span_begin, span_end = span
			observations.append(txtFileRead[span_begin:span_end])
		return observations

	def loadDictionary(self):
		dictionary = set()
		for file in self.vocFiles:
			with codecs.open(self.vocFiles[file], 'r', encoding="utf8", errors='ignore') as fp:
				for line in fp:
					row = line.split("\t")[1].split("|")
					for r in row:
						diseases = ""
						for word in r.split(" "):
							word = word.lower()
							#if word not in STOPWORDS and word not in SPECIALSTOPWORDS:
							#	diseases += word + " "
						dictionary.add(diseases.strip())
		return dictionary

	def loadFMObs(self, fmFile, obsFile):
		"""
		...
		:ŕeturn: Tuple of two dicts: fms: key is the file name, value is a list of tuples containing the FMs and sentence.
		(
			{
				"file name": [((FamilyMember, FamilySide), sentence),...]
			},
			{
				"file name": [(Observation, sentence),...]
			}
		)
		"""
		fms = {}
		obs = {}
		with codecs.open(fmFile, 'r', encoding="utf8", errors='ignore') as fp:
			for line in fp:
				print(line)
		return fms, obs
