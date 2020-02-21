from Preprocessing import nltkInitialize
from embeddings.Pipeline import runEmbeddingCreationPipeline
from Entity import createDefaultClasses, setObservationClasses, setFamilyMemberClasses

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

<<<<<<< HEAD
=======
			from embeddings.Pipeline import runPipeline
>>>>>>> 492484fe9e28c6e2a841296f2b5af8765095fe80
			runEmbeddingCreationPipeline(dataSettings)

			return dict(), dict()

		elif  method == "methodZZZ":
			return dict(), dict()
		return 	dict(), dict()

	def processTask2(show=False):
		# to do
		return 	None