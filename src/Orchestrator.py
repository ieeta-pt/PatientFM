from Preprocessing import nltkInitialize
from embeddings.Pipeline import runEmbeddingCreationPipeline
from Entity import ENTITY_CLASSES, createDefaultClasses, setObservationClasses, setFamilyMemberClasses, createTrueClasses
from RuleBased import RuleBased

# from models.BiLstmCRF.modelRunners import runModel, runModelDevelopment

from NejiAnnotator import runNejiSourcesCreation, readPickle
from models.ALBERT_BiLstmCRF.utils import ALBERTutils
from models.clinicalBERT.utils import clinicalBERTutils


class Orchestrator():
	def processTask1(files, XMLAnnotations, dictionaries, dataSettings, method=None, show=False):
		"""
		Method to handle with task 1 .
		:param files: dictionary containing the clinical reports (key: filename)
		:....

		returns tuple(dictionary containing family members (key: filename, value: list of tuples ((fm, fs), sentence),
					  dictionary containing observations (key: filename, value: list of tuples (obs, sentence)))
		"""
		if method == "silva":
			nltkInitialize(dataSettings["datasets"]["nltk_sources"])

			""" 
			Loads model runners according to the selected DL model (defined in settings.ini)
			"""
			if dataSettings["DLmodel"]["model"] == "biowordvec_bilstm":
				from models.Embedding_BiLstmCRF.modelRunners import runModel, runModelDevelopment, runModel_LoadAndTest
			elif dataSettings["DLmodel"]["model"] == "albert_bilstm":
				from models.ALBERT_BiLstmCRF.modelRunners import runModel, runModelDevelopment, runModel_LoadAndTest
			elif dataSettings["DLmodel"]["model"] == "clinicalbert_bilstm":
				from models.clinicalBERT_BiLstmCRF.modelRunners import runModel, runModelDevelopment, runModel_LoadAndTest
			elif dataSettings["DLmodel"]["model"] == "clinicalbert_linear":
				from models.clinicalBERT.modelRunners import runModel, runModelDevelopment, runModel_LoadAndTest

			""" 
			Embedding and neji sources creation is provided below if the user wants to create them from scratch.
			Files for these sources are already provided in the repository, hence these lines are commented
			"""
			# runEmbeddingCreationPipeline(dataSettings)
			# if dataSettings["neji"]["use_neji_annotations"] == "True":
			# 	runNejiSourcesCreation(dataSettings)
			# if dataSettings["neji"]["use_neji_annotations"] == "True":
			# 	albertUtils = ALBERTutils(dataSettings["ALBERT"]["model"], True)
			# 	runNejiSourcesCreation(dataSettings, modelType="albert", bertUtils=albertUtils)
			# if dataSettings["neji"]["use_neji_annotations"] == "True":
			# 	clinicalBERTUtils = clinicalBERTutils(True)
			# 	runNejiSourcesCreation(dataSettings, modelType="clinicalBERT", bertUtils=clinicalBERTUtils)

			"""
			Development Runner: train and validate using k-fold cross-validation. Uses only train dataset
			"""
			# predFamilyMemberDict, predObservationDict = runModelDevelopment(dataSettings, files, XMLAnnotations, cvFolds=5)

			"""
			Test Runner: trains in full train dataset, evaluates and generates output on full test dataset
			"""
			predFamilyMemberDict, predObservationDict = runModel(dataSettings, files, XMLAnnotations)

			return predFamilyMemberDict, predObservationDict

		elif method == "ja_rules":
			return RuleBased.processTask1(files)

		elif  method == "methodZZZ":
			return dict(), dict()
		return 	dict(), dict()

	def mergeResultsTask1(fmDocs, obsDocs):		
		"""
		Method to combine results for the different methods applied in task 1.
		Currently, we are using the rule-based system for family members identification and deep learning for observation extraction.
		:param fmDocs: dictionary containing the family members for the different methods. key: method used.
		:param obsDocs: dictionary containing the observations for the different methods. key: method used.
		returns tuple(dictionary containing family members (key: filename, value: list),
					  dictionary containing observations (key: filename, value: list))
		"""
		fmRes = dict()
		obsRes = dict()
		if "ja_rules" in fmDocs:
			fmRes = fmDocs["ja_rules"]
		else:
			fmRes = fmDocs[list(fmDocs.keys())[0]]
		obsRes = obsDocs[list(obsDocs.keys())[0]]
		return 	fmRes, obsRes

	def processTask2(show=False):
		# to do
		return 	None