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
		