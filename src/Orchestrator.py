from Preprocessing import nltkInitialize
from embeddings.Pipeline import runEmbeddingCreationPipeline
from Entity import ENTITY_CLASSES, createDefaultClasses, setObservationClasses, setFamilyMemberClasses, createTrueClasses
from RuleBased import RuleBased

# from models.BiLstmCRF.modelRunners import runModel, runModelDevelopment
#from models.Embedding_BiLstmCRF.modelRunners import runModel, runModelDevelopment
#from models.ALBERT_BiLstmCRF.modelRunners import runModel, runModelDevelopment
from models.clinicalBERT.modelRunners import runModel, runModelDevelopment

from NejiAnnotator import runNejiSourcesCreation, readPickle
from models.ALBERT_BiLstmCRF.utils import ALBERTutils


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

			r""" Embedding and neji sources creation is provided below if the user wants to create them from scratch.
			Files for these sources are already provided in the repository, hence these lines are commented"""
			# runEmbeddingCreationPipeline(dataSettings)
			# if dataSettings["neji"]["use_neji_annotations"] == "True":
			# 	runNejiSourcesCreation(dataSettings)
			# if dataSettings["neji"]["use_neji_annotations"] == "True":
			# 	albertUtils = ALBERTutils(dataSettings["ALBERT"]["model"], True)
			# 	runNejiSourcesCreation(dataSettings, "albert", bertUtils=albertUtils)

			# predFamilyMemberDict, predObservationDict = runModel(dataSettings, files, XMLAnnotations)
			predFamilyMemberDict, predObservationDict = runModelDevelopment(dataSettings, files, XMLAnnotations, cvFolds=5)

			return predFamilyMemberDict, predObservationDict
			return dict(), dict()

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