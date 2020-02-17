import glob
import codecs
#from stopWords import STOPWORDS, SPECIALSTOPWORDS

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

	def loadXMLAnnotations(self):
		#to do
		return None

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