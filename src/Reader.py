import glob
import codecs
#from stopWords import STOPWORDS, SPECIALSTOPWORDS

class Reader(object):
	def __init__(self, dataSettings):
		self.dataSetDir = dataSettings["datasets"]["files"]
		self.vocFiles = dataSettings["vocabulary"]

	def loadDataSet(self):
		filesContent = {}
		allFiles = glob.glob('{}*.{}'.format(self.dataSetDir, "txt"))
		for file in allFiles:
			fileName = file.split("/")[-1].split(".")[0]
			with codecs.open(file, 'r', encoding='utf8') as fp:
				filesContent[fileName] = fp.read()
		return filesContent

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