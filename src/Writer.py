class Writer():
	def writeTask1(resultFile, fmDocs, obsDocs):
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
			for familyMember, sideOfFamily in fmDocs[doc]:
				foutput.write("{}\tFamilyMember\t{}\t{}\n".format(doc, familyMember, sideOfFamily))
		
		for doc in sorted(obsDocs):
			#doc_1	Observation	Disease
			for disease in obsDocs[doc]:
				foutput.write("{}\tObservation\t{}\n".format(doc, disease))
		
		foutput.close()
		