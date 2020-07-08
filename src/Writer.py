class Writer():
	def writeTask1(resultFile, fmDocs, obsDocs, submission=False):
		"""
		Write output file for task 1.
		File content example:
			doc_1	FamilyMember	Father	NA
			doc_1	Observation	Disease
			...
		"""
		foutput = open(resultFile, "w")
				
		for doc in sorted(fmDocs):
			#doc_1	FamilyMember	Father	NA
			for fm, sentence in fmDocs[doc]:
				familyMember, sideOfFamily = fm
				if submission:
					foutput.write("{}\tFamilyMember\t{}\t{}\n".format(doc, familyMember, sideOfFamily))
				else:
					foutput.write("{}\tFamilyMember\t{}\t{}\t{}\n".format(doc, familyMember, sideOfFamily, sentence))
		
		for doc in sorted(obsDocs):
			#doc_1	Observation	Disease
			for disease, sentence in obsDocs[doc]:
				if submission:
					foutput.write("{}\tObservation\t{}\n".format(doc, disease))
				else:
					foutput.write("{}\tObservation\t{}\t{}\n".format(doc, disease, sentence))
		foutput.close()
		
	def writeTask2(resultFile, annotations):
		"""
		{
			"file name":[
				(familyMember, familySide, "LivingStatus", number 4 or 0)
				or
				(familyMember, familySide, "Observation", concept)
			]
		}
		"""
		foutput = open(resultFile, "w")
		for file in annotations:
			for ann in annotations[file]:
				annType = ann[2]
				familyMember = ann[0]
				sideOfFamily = ann[1]
				value = ann[3]
				if annType == "LivingStatus":
					foutput.write("{}\t{}\t{}\t{}\t{}\n".format(file, familyMember, sideOfFamily, annType, value))
				elif annType == "Observation":
					foutput.write("{}\t{}\t{}\t{}\t{}Non_Negated\n".format(file, familyMember, sideOfFamily, annType, value))
		foutput.close()