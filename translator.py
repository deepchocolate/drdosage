#!/usr/bin/env python3
# coding=utf8
from cla import *
import argparse, sys, json
class Translator(Mum):
	"""
	Does predictions given text input.
	"""
	def __init__(self, modelStream, inputStream, textColumn, outputStream):
		Mum.__init__(self, inputStream, textColumn, outputStream)
		self.load(modelStream)
		self._errors = []

	def load(self, modelStream):
		"""
		Load model to use.
		"""
		self.models = json.loads(modelStream.read())
		return self

	def goToNextMissing(self):
		"""
		Iterate until next untranscribed text
		"""
		while self.position() < self.nrRecords() and self._errors[self.position()] == False: next(self)
		return self

	
	def runModels(self):
		inc = 0
		for i,m in enumerate(self.models):
			print('Load model', i)
			Classifier.setObjectData(self, m['classifier'])
			Transcripts.setObjectData(self, m['transcripts'])
			print('Run model', i)
			inc += self.transcribe()
			print('% transcribed:',round(100*(1-sum(self._errors)/self.nrRecords()),2))
		print('Done')
		self.writeOutput()

	def transcribe (self):
		#self._ioData.rewind()
		#self.goToNextMissing()
		incorrect = 0
		while next(self) != False:
			if self.position() % 5000 == 0: print('Parsing record:', self.position() + 1)
			total = ''
			objects = ''
			val, pattern, error, p = self.transcribeCurrent()
			if val != False:
				total = sum(val.totals())
				objects = val.objects()
				error = 0
				#self._errors[self.position()-1] = False
			else:
				error = 1
				incorrect += 1
			if len(self._errors) >= self.position() + 1: self._errors[self.position()] = error
			else: self._errors += [error]
			self.setCol('total', total).setCol('manual', 0).setCol('pattern', pattern).setCol('error', error).setCol('prob', p).setCol('object', objects)
			self.setOutput()
		return incorrect

if __name__ == '__main__':
	ap = argparse.ArgumentParser(description="""Dr. Dose is a neural network based prediction engine tailored to 
	extract dosage information from pharmaceutial prescription texts in the Swedish Prescribed Drug Register.
	
	If data is read from a file this needs to be in a CSV format. If data is passed through standard in input data
	must consist of text sequences separated by linebreaks. Output file be provided in a CSV format with the same
	structure as input, but additional columns of pattern, total, probability, and error.""")
	ap.add_argument('model', metavar='prediction model', nargs=1, type=argparse.FileType('r'), help='Model to load and apply to [input] data or file.')
	ap.add_argument('-i', metavar='input data', type=argparse.FileType('r'), nargs='?', default=sys.stdin, help='File to read from, if omitted read from standard in.')
	ap.add_argument('-ic', metavar='text column', type=str, nargs=1, help='Input text column.')
	ap.add_argument('-o', metavar='output data', type=argparse.FileType('w'), nargs='?', default=sys.stdout, help='File to write to, if omitted output to standard out.')
	if len(sys.argv) > 1:
		args = ap.parse_args()
		t = Translator(args.model[0], args.i, args.ic[0], args.o)
		t.runModels()
	else:
		pass
