from .mum import *
from .classifier import *
from nltk.stem.snowball import SwedishStemmer
from fun.strman import *
from fun.num import *
from .transcript import *
import datetime, json
SKIPCHARS = ')(.?*=-+"!,\n'
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
		self._stemmer = SwedishStemmer()
		self._input = []
		self._output = []
	
	def filterAndSplitTranscript(self, transcript):
		return stripArguments(transcript.lower().replace('/','s')).split(' ')
		
	
	def filterAndSplitText(self, txt):
		return self.stemTokens(replaceNumbers(polish(txt.lower())))
	
	def stemTokens(self, text):
		o = []
		#for w in nltk.word_tokenize(text):
		for w in text.split(' '):
			if not w in SKIPCHARS: o += [self._stemmer.stem(w.lower())]
		return o
	
	def addIO(self, text, transcript):
		text = self.filterAndSplitText(text)
		if transcript == '': transcript = 's'+str(len(text))
		self.encode(text, transcript)
		transcript = self.filterAndSplitTranscript(transcript)
		if len(transcript) < len(text): transcript += ['s']*(len(text)-len(transcript))
		self.add(text, transcript)
		return self
	
	def getTrainingAttributes(self):
		data = {}
		data['meta'] = {'Name':'','Creator':'','Created':str(datetime.datetime.now()),'ATC':'','Description':''}
		data['model'] = self.getModelAttributes()#Classifier.getObjectData(self)
		data['transcripts'] = Transcripts.getObjectData(self)
		#self.models += [data]
		return data
	
	def save(self, outputStream):
		outputStream.write(json.dumps(self.getTrainingAttributes()))
		return self

class DiscreteTrainer(Trainer, Classifier):
	
	def parsePatterns(self, patternColumn, minExamples=6, maxExamples=30):
		print('Parsing... Including patterns with at least %s examples and limiting to the %s first examples' % (str(minExamples), str(maxExamples)))
		self.setTrainRunLimit(maxExamples=30, minExamples=6)
		i = 1
		n = 1
		included = 0
		while next(self) != False:
			txt = self.get(self._textColumn)
			txtSimple = simplify(replaceNumbers(txt))
			pattern = rmExtraSpace(self.get(patternColumn).lower().strip())
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
	
	def train(self, patternColumn, textColumn, neuronsLower=5, neuronsUpper=20):
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
				# Get pattern stripped of whitespace
				outputPattern = ''
				txt = rmExtraSpace(self.get(patternColumn).strip().lower())
				if len(txt) > 0:
					txt = txt.split(' ')
					#print(txt)
					for ch in txt: outputPattern += ch[0]
				if pattern != outputPattern: incorrect += 1
				#print(pattern+':'+outputPattern)
				#if value == False: incorrect += 1
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

class MarkovishTrainer(Trainer, Markovish):
	
	def __init__(self, trainStream=False, textColumn=False, optStream=False, optTextColumn=False, trailingWords=6,trailingOperators=4):
		Trainer.__init__(self, trainStream, textColumn, optStream, optTextColumn)
		Markovish.__init__(self, trailingWords, trailingOperators)
	
	def addIO(self, text, transcript):
		#print(nltk.word_tokenize('(a.b) 1:a a'))
		#quit()
		Trainer.addIO(self, text, transcript.replace(' ;', '').replace(' *',''))
		return self
	
	def getModelAttributes(self):
		return Markovish.getObjectData(self)

class OperatorMarkovishTrainer(Trainer, OperatorMarkovish):
	
	def __init__(self, trainStream=False, textColumn=False, optStream=False, optTextColumn=False, trailingWords=2,trailingOperators=2):
		Trainer.__init__(self, trainStream, textColumn, optStream, optTextColumn)
		OperatorMarkovish.__init__(self, trailingWords, trailingOperators)
	
	def addIO(self, text, transcript):
		text = self.filterAndSplitText(text)
		#if transcript == '': transcript = 's'+str(len(text))
		#transcript = self.filterAndSplitTranscript(transcript)
		# Add plusses between operators (the default mode)
		newTranscript, newText = [], []
		transcript = self.filterAndSplitTranscript(transcript)
		#if len(transcript) < len(text): transcript += ['s']*(len(text)-len(transcript))
		i = 0
		for x in transcript:
			newText += [text[i]]
			if len(newTranscript) == 0:
				newTranscript += [x]
				i += 1
			elif newTranscript[len(newTranscript)-1][0] in 'qofts' and x[0] in 'qofts':
				newTranscript += ['+',x]
				i += 1
			# If operator
			else: newTranscript += [x]
		#transcript = ' '.join(newTranscript)
		#transcript = newTranscript
		#print(newTranscript)
		self.add(newText, newTranscript)
		return self

	def getModelAttributes(self):
		return OperatorMarkovish.getObjectData(self)
