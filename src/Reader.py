import glob
import codecs
#from stopWords import STOPWORDS, SPECIALSTOPWORDS

import xml.etree.ElementTree as ET

class Reader(object):
	def __init__(self, dataSettings):
		self.dataSetDir = dataSettings["datasets"]["files"]
		self.vocFiles = dataSettings["vocabulary"]

	def loadDataSet(self, cleaning=False):
		filesContent = {}
		allFiles = glob.glob('{}*.{}'.format(self.dataSetDir, "txt"))
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
		allFiles = glob.glob('{}*.{}'.format(self.dataSetDir, "xml"))
		for file in allFiles:
			fileName = file.split("/")[-1].split(".")[0]
			xmlFilesContent[fileName] = {}
			fileTree = ET.parse(file)
			xmlFilesContent, shitcount = self.extractXMLEntity(fileTree, fileName, xmlFilesContent, dataset)
			xmlFilesContent = self.extractXMLRelation(fileTree, fileName, xmlFilesContent)
		return xmlFilesContent

	# def extractXMLEntity(self, fileElementTree, fileName, filesContent):
	# 	"""
	# 	Searches the ElementTree for annotations regarding Entities and updates the filesContent dict with extracted information
	# 	returns filesContent
	# 	"""
	#
	# 	for entity in fileElementTree.iter('entity'):
	# 		id             = entity.find('id').text
	# 		span           = entity.find('span').text
	# 		annotationType = entity.find('type').text
	# 		properties 	   = entity.find('properties')
	#
	# 		if annotationType == "Age":
	# 			ageType = properties.find('AgeType').text
	# 			filesContent[fileName].update({id: {'span': span, 'type': annotationType, "ageType": ageType}})
	#
	# 		elif annotationType == "FamilyMember":
	# 			count 		   = properties.find('Count').text
	# 			familyRelation = properties.find('Relation').text
	# 			familySide     = properties.find('SideOfFamily').text
	# 			filesContent[fileName].update({id: {'span': span, 'type': annotationType, "familyRelation": familyRelation, "count": count, "familySide": familySide}})
	#
	# 		elif annotationType == "Observation":
	# 			negation  = properties.find('Negation').text
	# 			certainty = properties.find('Certainty').text
	# 			filesContent[fileName].update({id: {'span': span, 'type': annotationType, "negation": negation, "certainty": certainty}})
	#
	# 		elif annotationType == "LivingStatus":
	# 			alive   = properties.find('Alive').text
	# 			healthy = properties.find('Healthy').text
	# 			filesContent[fileName].update({id: {'span': span, 'type': annotationType, "alive": alive, "healthy": healthy}})
	# 	return filesContent

	def extractXMLEntity(self, fileElementTree, fileName, xmlFilesContent, dataset):
		"""
		Searches the ElementTree for annotations regarding Entities and updates the filesContent dict with extracted information
		returns xmlFilesContent
		"""

		for entity in fileElementTree.iter('entity'):
			id             = entity.find('id').text
			span           = entity.find('span').text
			annotationType = entity.find('type').text
			properties 	   = entity.find('properties')

			if annotationType == "Age":
				ageType = properties.find('AgeType').text
				xmlFilesContent[fileName].update({id: {'span': span, 'type': annotationType, "ageType": ageType}})

			elif annotationType == "FamilyMember":
				count 		   = properties.find('Count').text
				familyRelation = properties.find('Relation').text
				familySide     = properties.find('SideOfFamily').text
				xmlFilesContent[fileName].update({id: {'span': span, 'type': annotationType, "familyRelation": familyRelation, "count": count, "familySide": familySide}})

			elif annotationType == "Observation":
				negation  = properties.find('Negation').text
				certainty = properties.find('Certainty').text
				xmlFilesContent[fileName].update({id: {'span': span, 'type': annotationType, "negation": negation, "certainty": certainty}})

			elif annotationType == "LivingStatus":
				alive   = properties.find('Alive').text
				healthy = properties.find('Healthy').text
				xmlFilesContent[fileName].update({id: {'span': span, 'type': annotationType, "alive": alive, "healthy": healthy}})
		return xmlFilesContent

	def extractXMLRelation(self, fileElementTree, fileName, xmlFilesContent):
		"""
		Searches the ElementTree for annotations regarding Relations and updates the filesContent dict with extracted information
		returns xmlFilesContent
		"""

		for relation in fileElementTree.iter('relation'):
			id             = relation.find('id').text
			annotationType = relation.find('type').text
			properties 	   = relation.find('properties')
			familyMembers  = properties.find('FamilyMembers').text
			sub_properties = [sub_property.text for sub_property in properties.findall('Properties')]
			xmlFilesContent[fileName].update({id: {'type': annotationType, "familyMembers": familyMembers, "Properties": sub_properties}})
		return xmlFilesContent

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