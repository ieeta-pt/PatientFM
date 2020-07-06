import argparse
import sys
import configparser
import time
from Reader import Reader
from Writer import Writer
from Orchestrator import Orchestrator

def help(show=False):
	parser = argparse.ArgumentParser(description="")
	configs = parser.add_argument_group('Global settings', 'This settings are related with the location of the files and directories.')
	configs.add_argument('-s', '--settings', dest='settings', \
                        type=str, default="settings.ini", \
                        help='The system settings file (default: settings.ini)')	
	configs.add_argument('-p', '--showprints', default=False, action='store_true', \
							help='When active it show some parts of the processing during the executions (default: False)')
	configs.add_argument('-c', '--cleaning', default=False, action='store_true', \
							help='When active it cleans the clinical reports (default: False)')
	configs.add_argument('-m', '--method', dest='method', type=str, default='',\
                        help='The method used to process.')
	configs.add_argument('-ds', '--dataset', dest='dataset', type=str, default='train',\
                        help='The dataset used to process.')
	configs.add_argument('-r', '--read', dest='dataset', type=str, default='train',\
                        help='Read annotations from file (Usefull for task 2)')
	
	executionMode = parser.add_argument_group('Execution Mode', 'Flags to select the execution mode!')
	executionMode.add_argument('-t1', '--first', default=False, action='store_true', \
							help='In this mode, the script will execute the first subtask of the challenge (default: False)')
	executionMode.add_argument('-t2', '--second', default=False, action='store_true', \
							help='In this mode, the script will execute the second subtask of the challenge (default: False)')
	if show:
		parser.print_help()
	return parser.parse_args()

def readSettings(settingsFile):
	configuration = configparser.ConfigParser()
	configuration.read(settingsFile)
	if not configuration:
		raise Exception("The settings file was not found!")
	return configuration

def getMethod(settings, argsMethod):
	if len(argsMethod) >0:
		return [argsMethod]
	methods = []
	for method in settings["methods"]:
		if settings["methods"][method] == "True":
			methods += [method]
	return methods

def main():
	args = help()
	settings = readSettings(args.settings)
	if  not args.first and \
		not args.second:
		print("Nothing to do, please define the method to process!")
		help(show=True)
		exit()

	methods = getMethod(settings, args.method)
	if  len(methods) == 0:
		print("Nothing to do, please select the execution mode!")
		help(show=True)
		exit()
	fmDocsRes = {}
	obsDocsRes = {}
	if args.first:
		print("Execute the first subtask")
		reader = Reader(dataSettings=settings, corpus=args.dataset)
		filesRead = reader.loadDataSet(cleaning=args.cleaning)
		fmDocs = {}
		obsDocs = {}
		for method in methods:
			fm, obs = Orchestrator.processTask1(files 			= filesRead,
												XMLAnnotations 	= reader.loadXMLAnnotations(filesRead),
												dictionaries	= reader.loadDictionary(),
												dataSettings    = settings,
												method			= method,
								    			show 			= args.showprints)
			fmDocs[method] = fm
			obsDocs[method] = obs
		fmDocsRes, obsDocsRes = Orchestrator.mergeResultsTask1(fmDocs, obsDocs)
		Writer.writeTask1(resultFile 	= settings["results"]["task1"], 
					 	  fmDocs 		= fmDocsRes, 
					 	  obsDocs	 	= obsDocsRes)

	if args.second:
		print("Execute the second subtask")
		if args.read:
			fmDocsRes, obsDocsRes = Reader.loadFMandObs()
		if len(fmDocsRes) == 0 or len(obsDocsRes) == 0:
			print("No observations or family members available for this task!")
		else:
			print("to do")
main()