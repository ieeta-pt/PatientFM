from Entity import createDefaultClasses, setObservationClasses, setFamilyMemberClasses

class Orchestrator():
	def processTask1(files, XMLAnnotations, dictionaries, method=None, show=False):
		"""
		Method to handle with task 1 .
		:param files: dictionary containing the clinical reports (key: filename)
		:....

		returns tuple(dictionary containing family members (key: filename, value: list),
					  dictionary containing observations (key: filename, value: list))
		"""
		# to do
		if method == "silva":
			classesDict = createDefaultClasses(files)
			classesDict = setObservationClasses(classesDict, files, XMLAnnotations)
			classesDict = setFamilyMemberClasses(classesDict, files, XMLAnnotations)
			# for fileName in files:
			# 	print(fileName)
			# 	print(classesDict[fileName])
			return dict(), dict()

		elif  method == "methodZZZ":
			return dict(), dict()
		return 	dict(), dict()

	def processTask2(show=False):
		# to do
		return 	None