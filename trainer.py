#!/usr/bin/env python3
# coding=utf8
import operator
#import nltk
# Run import nltk; nltk.download('punkt')
import os, sys, argparse, json, datetime, numpy as np, re, csv
### Helper functions: isNumeric() and rmExtraSpace()
#from fun import rmExtraSpace, simplify, replaceNumbers
# Classes
from cla import *
from fun import *

if __name__ == '__main__':
	ap = argparse.ArgumentParser(description="""Dr. Dose is a neural network based prediction engine tailored to 
	extract dosage information from pharmaceutial prescription texts in the Swedish Prescribed Drug Register.
	
	If model is unspecified the default action is to train data and output. In/outfile needs to be in a CSV-format.""")
	ap.add_argument('-n', metavar='neurons', nargs='+', default=[5, 5],
		help='Set no. neurons in hidden layer. Either a single number or an upper and lower limit. Defaults to 5. During optimization the number yielding the best fit will be stored in the model.')
	ap.add_argument('-t', metavar='optimization file', default=False, type=argparse.FileType('r'), nargs='?',
		help='If provided, this file will be used when scaling the hidden layer, otherwise training file(s) will be used.')
	ap.add_argument('-ttc', metavar='optimization text column', default=False, nargs='?', help='Name or number of column where input texts resides if optimization file is provided.')
	ap.add_argument('-tc', metavar='text column', nargs=1, help='Name or number of column where input texts resides.')
	ap.add_argument('-pc', metavar='pattern column', nargs=1, type=str, help='Name or number of column where patterns to input texts resides.')
	ap.add_argument('-o',metavar='output file', type=argparse.FileType('w'), nargs='?', default=sys.stdout, help='File to write to, if omitted output to standard out.')
	ap.add_argument('training',metavar='training file(s)', type=argparse.FileType('r'), nargs='+', 
		help='File(s) to read from. If several files are provided, one model will be fitted for each training data and applied consecutively.')
	if len(sys.argv) > 1:
		args = ap.parse_args()
		nL, nU = args.n if len(args.n)==2 else [args.n[0], args.n[0]]
		t = Trainer(False, args.tc[0], args.t, args.ttc)
		for train in args.training:
			optimal, accuracy = t.setInput(train, args.tc[0]).parsePatterns(args.pc[0]).train(args.pc[0],args.tc[0], int(nL), int(nU))
			print('Maximum accuracy reached with '+str(optimal)+' neurons: '+str(accuracy)+'% predicted correctly.')
			t.saveModel()
		t.save(args.o)
	else:
		dire = '../csv/'
		a = MarkovishTrainer(open(dire+'dispensations_training_sample.csv'),'doser')
		b = OperatorMarkovishTrainer(open(dire+'dispensations_training_sample.csv'),'doser')
		i = 1
		while next(a):
			print('Record: ' + str(i))
			print('\tText: ' + a.get('doser'))
			print('\tTranscript: ' + a.get('pattern'))
			a.addIO(a.get('doser'),a.get('pattern'))
			if a.get('pattern') != '': b.addIO(a.get('doser'),a.get('pattern'))
			#a.encode(a.get('doser'),a.get('pattern'))
			i += 1
		b.fit()
		b.save(open('myOperatorModel2.model','w'))
		#quit()
		a.fit()
		a.save(open('myModel2.model','w'))
		quit()
		a.rewind()
		N = 0
		correctN = 0
		errors = ''
		while next(a):
			N += 1
			cleanInput = ''
			if a.get('pattern') != '': cleanInput = stripArguments(a.get('pattern').lower().replace(' ;','').replace(' *','').replace('/','s'))
			cleanOutput = ' '.join(a.predictSequence(a.get('doser')))
			#print(f.get('doser')+': ' + cleanInput + ' : ' + cleanOutput)
			if cleanInput.replace(' ','').rstrip('s') == cleanOutput.replace(' ', '').rstrip('s'): correctN += 1
			else: errors += a.get('doser')+': ' + cleanInput + ' : ' + cleanOutput + "\n"
		print(correctN,N)
		quit()
		# Example of coded use
		trainingData = '../csv/meds_100_train.csv'
		textColumn = 'doser'
		patternColumn = 'pattern'
		neuronsLowerLimit = 10
		neuronsUpperLimit = 20
		t = Transcripts()
		s = Markovish(6,4, False)
		#t.addTrancript('one pill per day week 1 2 pills per day weeks two','q1 opill s2 tw t q opill s2 tw t2')
		#t._funs['q']('two','2')
		f = FileIO(open(dire+'dispensations_training_sample.csv'))
		output = 'id,time,dose\n'
		i = 0;
		while next(f) != False:
			print('Record: '+str(i)+' '+f.get('pattern'))
			t.encode(f.get('doser'), f.get('pattern'))
			if f.get('pattern') != '':
				pattern = stripArguments(f.get('pattern').lower().replace(' ;','').replace(' *',''))
				s.add(f.get('doser'), pattern)
				for x in t.decode(f.get('doser'),stripArguments(f.get('pattern').lower())):
					output += f.get('id') + ','+str(x.timePoint()) + ','+str(x.total())+'\n'
			#print(stripArguments(f.get('pattern')))
		#print(str(i)+': '+f.get('doser'))
		i += 1
		open(dire+'testoutput.csv','w').write(output)
		fit = s.fit()
		s.save(open('../seqModel.model','w'))
		f.rewind()
		correctN = 0
		N = 0
		errors = ''
		s = Markovish()
		s.load(open('../seqModel.model'))
		while next(f) != False:
			if f.get('pattern') != '':
				N += 1
				cleanInput = stripArguments(f.get('pattern').lower().replace(' ;','').replace(' *','').replace('/','s'))
				cleanOutput = ' '.join(s.predictSequence(f.get('doser')))
				print(f.get('doser')+': ' + cleanInput + ' : ' + cleanOutput)
				if cleanInput.replace(' ','').rstrip('s') == cleanOutput.replace(' ', '').rstrip('s'): correctN += 1
				else: errors += f.get('doser')+': ' + cleanInput + ' : ' + cleanOutput + "\n"
		print('ERRORS')
		print(errors)
		print(correctN, N)
		wrds = ''
		# Load object
		#t = Trainer(open(trainingData), textColumn)
		# Parse trainingdata and fit the model
		#optimal,accuracy = t.parsePatterns(patternColumn).train(patternColumn,textColumn,neuronsLowerLimit,neuronsUpperLimit)
		# Save the model
		#t.saveModel().save(open('myModel.model','w'))
		pass
# Fix bug: leading/trailing blanks in patterns
