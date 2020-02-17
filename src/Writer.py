class Writer():
	def writeTask1(resultFile, fh, obs):
		foutput = open(resultFile, "w")
				
		for doc in sorted(fh):
			#doc_1	FamilyMember	Father	NA
			for familyMember, sideOfFamily in fh[doc]:
				foutput.write(doc+"\tFamilyMember\t"+familyMember+"\t"+sideOfFamily+"\n")
		
		for doc in sorted(obs):
			#doc_1	Observation	Disease
			for disease in obs[doc]:
				foutput.write(doc+"\tObservation\t"+disease+"\n")
		
		foutput.close()
		