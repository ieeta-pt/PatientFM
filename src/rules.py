deadWords = [
	"died", "passed away", "death", "deceased", "killed", "murdered"
]
patientWordList = [
	"patient's","patient","patients"
]
malePatientWordList = [
	'he', "he'd", "hed", "he'll", "he's", "hes", 'him', 'himself', 'his'
]
femalePatientWordList = [
	'she', "she'd", "she'll", "she's", 'shed', 'shes', "her", 'hers', 'herself'
]

#Simple rules to detect family members
femaleFMPrivate = ["Mother", "Sister", "Daughter",]
maleFMPrivate = ["Father",  "Brother",  "Son",]
unisexFM = ["Child", "Sibling",]
commonFM = unisexFM + maleFMPrivate + femaleFMPrivate
femaleFMwithSide = [
	"Aunt Maternal", "Aunt Paternal", "Grandmother Maternal", "Grandmother Paternal", 
]
femaleFMwithSideInverted = [
	"Maternal Aunt", "Paternal Aunt", "Maternal Grandmother", "Paternal Grandmother", 
]
maleFMwithSide = [
	"Uncle Maternal", "Uncle Paternal", "Grandfather Maternal", "Grandfather Paternal",
]
maleFMwithSideInverted = [ 
	"Maternal Uncle", "Paternal Uncle", "Maternal Grandfather", "Paternal Grandfather",
]
unisexFMwithSide = [
	"Cousin Maternal", "Grandparent Maternal", "Cousin Paternal", "Grandparent Paternal",
]
unisexFMwithSideInverted = [
	"Maternal Cousin", "Maternal Grandparent", "Paternal Cousin", "Paternal Grandparent",
]
commonFMwithSide = femaleFMwithSide + maleFMwithSide + unisexFMwithSide
commonFMwithSideInverted = femaleFMwithSideInverted + maleFMwithSideInverted + unisexFMwithSideInverted

familyMembers = commonFM + commonFMwithSide
familyMembersInverted = commonFM + commonFMwithSideInverted
femaleFM = femaleFMPrivate + femaleFMwithSide + femaleFMwithSideInverted
maleFM = maleFMPrivate + maleFMwithSide + maleFMwithSideInverted
unisexFM = unisexFM + unisexFMwithSide + unisexFMwithSideInverted

#For sex
femaleFMSex = femaleFMPrivate + ["Aunt", "Grandmother", "Geat Aunt"]
maleFMSex = maleFMPrivate + ["Uncle", "Grandfather", "Geat Uncle"]
unisexFMSex = unisexFM + ["Cousin", "Grandparent"]


#("", "", ""),
#("Word in text", "Family Member", "Side of family")
simpleTermsSecond = [
	("father","Father", "NA"), ("dad","Father", "NA"), ("dady","Father", "NA"),
	("mother", "Mother", "NA"), ("mum","Mother", "NA"), ("mom","Mother", "NA"),
	("sister", "Sister", "NA"), ("sisters", "Sister", "NA"), 
	("brother","Brother", "NA"), ("brothers","Brother", "NA"), 
	("daughter", "Daughter", "NA"), ("daughters","Daughter", "NA"),
	("son", "Son", "NA"), 
	#("child", "Child", "NA"), ("children", "Child", "NA"), 
	("sibling", "Sibling", "NA"), ("siblings", "Sibling", "NA"), 

	("uncle", "Uncle", "NA"),
	("aunt", "Aunt", "NA"),
	("cousin", "Cousin", "NA"),("cousins", "Cousin", "NA"),
	("grandmother", "Grandmother", "NA"),
	("grandfather", "Grandfather", "NA"),
	("grandparent", "Grandparent", "NA"), ("grandparents", "Grandparent", "NA"),

	#with errors
	("bothers","Brother", "NA"), ("bother","Brother", "NA"),
]
#("", "", "", [""], [""], True/False),
#("Word in text", "Family Member", "Side of family", ["words","before"], ["words","after"], Accept/Reject)
complexTermsFirst = [

	("dad", "Father", "NA", [], [],  True),
	("dady", "Father", "NA", [], [],  True),
	("mom", "Mother", "NA", [], [],  True),

	("child", "Child", "NA", ["patient"], [],  True),
	("child", "Child", "NA", ["patient's"], [],  True),
	("children", "Child", "NA", ["patient"], [],  True),
	("children", "Child", "NA", ["patient's"], [],  True),

	#True
	("daughter", "Daughter", "NA", ["patient"], [],  True),
	("daughter", "Daughter", "NA", ["patient's"], [],  True),
	("daughters", "Daughter", "NA", ["patient"], [],  True),
	("daughters", "Daughter", "NA", ["patient's"], [],  True),
	("son", "Son", "NA", ["patient"], [],  True),
	("son", "Son", "NA", ["patient's"], [],  True),
	("sons", "Son", "NA", ["patient"], [],  True),
	("sons", "Son", "NA", ["patient's"], [],  True),
	("sibling", "Son", "NA", ["patient"], [],  True),
	("sibling", "Son", "NA", ["patient's"], [],  True),
	("siblings", "Son", "NA", ["patient"], [],  True),
	("siblings", "Son", "NA", ["patient's"], [],  True),


	("brother", "Brother", "NA", ["patient's"], [],  True),
	("sister", "Sister", "NA", ["patient's"], [],  True),
	("brothers", "Brother", "NA", ["patient's"], [],  True),
	("sisters", "Sister", "NA", ["patient's"], [],  True),
	#This two increase the recall but decrease the precision because the words that i removed between
	("father", "Father", "NA", ["patient's"], [],  True),
	("mother", "Mother", "NA", ["patient's"], [],  True),
	("brother", "Brother", "NA", ["patient"], [],  True),
	("sister", "Sister", "NA", ["patient"], [],  True),
	("father", "Father", "NA", ["patient"], [],  True),
	("mother", "Mother", "NA", ["patient"], [],  True),

	("grandmother", "Grandmother", "Maternal", ["maternal"], [], True),
	("grandmother", "Grandmother", "Paternal", ["paternal"], [], True),
	("grandfather", "Grandfather", "Maternal", ["maternal"], [], True),
	("grandfather", "Grandfather", "Paternal", ["paternal"], [], True),

	#Only if there is no grandmother or grandfather
	("grandparents", "Grandparent", "Maternal", ["patient's", "maternal"], [], True),
	("grandparents", "Grandparent", "Paternal", ["patient's", "paternal"], [], True),

	#Aunt and Uncle
	("aunt", "Aunt", "Maternal", ["maternal"], [], True),
	("aunt", "Aunt", "Paternal", ["paternal"], [], True),
	#("antn", "Aunt", "Maternal", ["maternal"], [], True),
	#("antn", "Aunt", "Paternal", ["paternal"], [], True),
	("aunts", "Aunt", "Maternal", ["maternal"], [], True),
	("aunts", "Aunt", "Paternal", ["paternal"], [], True),
	("uncle", "Uncle", "Maternal", ["maternal"], [], True),
	("uncle", "Uncle", "Paternal", ["paternal"], [], True),
	("uncles", "Uncle", "Maternal", ["maternal"], [], True),
	("uncles", "Uncle", "Paternal", ["paternal"], [], True),
	
	#Cousin
	("cousin", "Cousin", "Paternal", ["paternal"], [], True),
	("cousin", "Cousin", "Maternal", ["maternal"], [], True),
	("cousins", "Cousin", "Paternal", ["paternal"], [], True),
	("cousins", "Cousin", "Maternal", ["maternal"], [], True),
	("son", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("son", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("son", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("son", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("sons", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("sons", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("sons", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("sons", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("daughter", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("daughter", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("daughter", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("daughter", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("daughters", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("daughters", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("daughters", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("daughters", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("sibling", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("sibling", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("sibling", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("sibling", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("child", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("child", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("child", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("child", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("siblings", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("siblings", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("siblings", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("siblings", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),
	("children", "Cousin", "Maternal", ["maternal", "aunt's"], [], True),
	("children", "Cousin", "Paternal", ["paternal", "aunt's"], [], True),
	("children", "Cousin", "Maternal", ["maternal", "uncle's"], [], True),
	("children", "Cousin", "Paternal", ["paternal", "uncle's"], [], True),

	
	#("son", "Cousin", "", ["maternal", "great", "aunt's"], [],  False),
	#("son", "Cousin", "", ["paternal", "great", "aunt's"], [],  False),
	#("daughter", "Cousin", "", ["maternal", "great", "aunt's"], [],  False),
	#("daughter", "Cousin", "", ["paternal", "great", "aunt's"], [],  False),
	("son", "2nd Cousin", "", ["great", "aunt's"], [],  False),
	("daughter", "2nd Cousin", "", ["great", "aunt's"], [],  False),
	("son", "2nd Cousin", "", ["great", "aunt"], [],  False),
	("daughter", "2nd Cousin", "", ["great", "aunt"], [],  False),


	("aunt", "Geat Aunt", "", ["great"], [], False),
	("uncle", "Geat Uncle", "", ["great"], [], False),

	#TO DO Half--

	#Not sure but pretty sure
	("baby", "Child", "", ["father"], [],  False),
	("baby", "Child", "", ["mother"], [],  False),
]

negationWords = ["no", "not"]

essentialWords = ["patient's"]

noPatientRelatedWordsList = ["husband","husband's", "wife", "wife's", "partner", "bride", 
	"companion", "spouse", "roommate", "mate", "consort", "boyfriend", "partner's", "girlfriend"]

unlockPatientRelatedWordsList = [("husband", "her")]

prenoms = ["mr", "mrs", "ms", "miss"]#maybe more

relations = [
	#fm1, fm2, relation to patient. Ex: uncle, son, patient's cousin
	#Father, Mother, Parent, Sister, Brother, Daughter, Son, Child,
	#Grandmother, Grandfather, Grandparent, Cousin, Sibling, Aunt, Uncle
	("Sister", "Son", None),#"Nephew"
	("Sister", "Daughter", None),#"Nephew"
	("Sister", "Child", None),#"Nephew"
	
	("Brother", "Son", None),#"Nephew"
	("Brother", "Daughter", None),#"Nephew"
	("Brother", "Child", None),#"Nephew"

	("Cousin", "Son", None),
	("Cousin", "Daughter", None),
	("Cousin", "Child", None),

	("Son", "Daughter", None),
	("Son", "Son", None),
	("Son", "Child", None),
	("Son", "Cousin", None),

	("Daughter", "Daughter", None),
	("Daughter", "Son", None),
	("Daughter", "Child", None),
	("Daughter", "Cousin", None),

	("Aunt", "Daughter", "Cousin"),
	("Aunt", "Son", "Cousin"),
	("Aunt", "Child", "Cousin"),
	("Aunt", "Cousin", None),
	
	("Uncle", "Daughter", "Cousin"),
	("Uncle", "Son", "Cousin"),
	("Uncle", "Child", "Cousin"),
	("Uncle", "Cousin", None),

	("Mother", "Brother", "Uncle"),
	("Mother", "Sister", "Aunt"),
	("Mother", "Son", "Brother"),
	("Mother", "Daughter", "Sister"),
	("Mother", "Child", "Sibling"),

	("Father", "Brother", "Uncle"),
	("Father", "Sister", "Aunt"),
	("Father", "Son", "Brother"),
	("Father", "Daughter", "Sister"),
	("Father", "Child", "Sibling"),
]

def getSimpleTermsSecond():
	listOfTerms = []
	for (x, y, z) in simpleTermsSecond:
		listOfTerms += [x.lower()] + [y.lower()] + [z.lower()]	
	return listOfTerms

def getComplexTermsSecond():
	listOfTerms = []
	#("daughter", "Daughter", "NA", ["patient"], [],  True),
	for (x, y, z, list1, list2, boolean) in complexTermsFirst:
		listOfTerms += [x.lower()] + [y.lower()] + [z.lower()]
		for a in list1:
			listOfTerms += [a.lower()]
		for a in list2:
			listOfTerms += [a.lower()]
	return listOfTerms

wordsToKeep =  set(prenoms + noPatientRelatedWordsList + essentialWords + negationWords +\
 getSimpleTermsSecond() + getComplexTermsSecond())
