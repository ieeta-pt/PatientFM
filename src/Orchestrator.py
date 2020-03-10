from Preprocessing import nltkInitialize
from embeddings.Pipeline import runEmbeddingCreationPipeline
from Entity import ENTITY_CLASSES, createDefaultClasses, setObservationClasses, setFamilyMemberClasses, createTrueClasses
from RuleBased import RuleBased

# from models.BiLstmCRF.modelRunners import runModel, runModelDevelopment
from models.Embedding_BiLstmCRF.modelRunners import runModel, runModelDevelopment

class Orchestrator():
	def processTask1(files, XMLAnnotations, dictionaries, dataSettings, method=None, show=False):
		"""
		Method to handle with task 1 .
		:param files: dictionary containing the clinical reports (key: filename)
		:....

		returns tuple(dictionary containing family members (key: filename, value: list),
					  dictionary containing observations (key: filename, value: list))
		"""
		# to do
		if method == "silva":
			nltkInitialize(dataSettings["datasets"]["nltk_sources"])
			# classesDict = createDefaultClasses(files)
			# classesDict = setObservationClasses(classesDict, files, XMLAnnotations)
			# classesDict = setFamilyMemberClasses(classesDict, files, XMLAnnotations)
			# classesDict = createTrueClasses(files, XMLAnnotations)

			runEmbeddingCreationPipeline(dataSettings)
			# predFamilyMemberDict, predObservationDict = runModel(dataSettings)

			predFamilyMemberDict, predObservationDict = runModelDevelopment(dataSettings, files, XMLAnnotations, cvFolds=5)
			return predFamilyMemberDict, predObservationDict
			# return dict(), dict()

		elif method == "ja_rules":
			return RuleBased.processTask1(files)

		elif  method == "methodZZZ":
			return dict(), dict()
		return 	dict(), dict()

	def mergeResultsTask1(fmDocs, obsDocs):
		fmRes = dict()
		obsRes = dict()
		#for method in fmDocs:
		#...
		#no right, in progress
		fmRes = fmDocs[list(fmDocs.keys())[0]]
		obsRes = obsDocs[list(obsDocs.keys())[0]]
		return 	fmRes, obsRes

	def processTask2(show=False):
		# to do
		return 	None