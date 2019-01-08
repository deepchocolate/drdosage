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
from fun import rmExtraSpace,simplify,replaceNumbers

class Trainer(Mum):
	def __init__(self, trainStream=False, textColumn=False, optStream=False, optTextColumn=False):
		"""
		trainStream: File with training data.
		textColumn: Column number or name of text.
		patternColumn: Column number or name of pattern.
		modelFile: File to save the model in.
		optStream: If optimizing using text in trainFile instead.
		optTextColumn: Column of text when using optFile.
		"""
		Mum.__init__(self, trainStream, textColumn)
		#self.parsePatterns(patternColumn);
		# If optStream is passed, then load training data separately.
		self._optTextColumn = optTextColumn
		self._optStream = optStream
		#self.train(self._textColumn,5,10)
		#self.saveModel().save('tests')
	
	def setInput(self, trainStream, textColumn):
		FileIO.__init__(self, trainStream, textColumn)
		return self
	
	def parsePatterns(self, patternColumn, minExamples=6, maxExamples=30):
		print('Parsing... Including patterns with at least %s examples and limiting to the %s first examples' % (str(minExamples), str(maxExamples)))
		self.setTrainRunLimit(maxExamples=30, minExamples=6)
		i = 1
		n = 1
		included = 0
		while next(self) != False:
			txt = self.get(self._textColumn)
			txtSimple = simplify(replaceNumbers(txt))
			pattern = self.get(patternColumn).lower()
			if pattern not in ['-', '']:
				try:
					transcript = self.parseInstruction(txt, pattern)
					self.addPattern(transcript, txtSimple)
					included += 1
				except BaseException:
					print('Record %s does not parse with pattern %s for input: %s' % (str(i), pattern, txt))
				i += 1
			n += 1
		print('Records read: %s\nRecords parsed: %s\n\tSuccesses: %s\n\tErrors: %s\n\tPercent success: %s' % (str(n), str(i), str(included), str(i-included), str(round(100*included/i,2))))
		self.updateIO()
		print('Patterns parsed:')
		sort = [(k, self.patternCounts[k]) for k in sorted(self.patternCounts, key=self.patternCounts.get, reverse=True)]
		for x in sort:
			print('%s: %s' % x)
		return self
	
	def train(self, textColumn, neuronsLower=5, neuronsUpper=20):
		# Load other to optimize size of the hidden layer on.
		if self._optStream: FileIO.__init__(self, self._optStream, self._optTextColumn)
		optimal = 0
		accuracy = 0
		maxAccuracy = 0
		synapse = []
		for neurons in iter(range(neuronsLower, neuronsUpper+1)):
			self.setNrHiddenNeurons(neurons).fit(True)
			print('Predicting on training data...')
			self.rewind()
			incorrect = 0
			i = 0
			while next(self) != False:
				if i % 5000 == 0: print('Parsing record:',i + 1)
				value, pattern, error, prob = self.transcribeCurrent()
				if value == False: incorrect += 1
				i += 1
			accuracy = 1
			if incorrect <= i: accuracy = 1 - incorrect/i
			if accuracy > maxAccuracy:
				optimal = neurons
				maxAccuracy = accuracy
				synapse = self.synapse
			print('% transcribed:',round(100*accuracy,2))
		self.setNrHiddenNeurons(optimal)
		self.synapse = synapse
		return optimal, maxAccuracy
	
	def saveModel(self):
		data = {}
		data['classifier'] = Classifier.getObjectData(self)
		data['transcripts'] = Transcripts.getObjectData(self)
		self.models += [data]
		return self
	
	def save(self, outputStream):
		outputStream.write(json.dumps(self.models) if len(self.models) > 0 else '')
		return self

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
			t.setInput(train, args.tc[0]).parsePatterns(args.pc[0]).train(args.tc[0], int(nL), int(nU))
			t.saveModel()
		t.save(args.o)
	else:
		pass
