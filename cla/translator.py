from .mum import *
from .transcripts import *
from .classifier import *
from fun.strman import *
from fun.num import *
from nltk.stem.snowball import SwedishStemmer
import json
class Translator(Mum):
	"""
	Does predictions given text input.
	"""
	def __init__(self, modelStream, inputStream, textColumn, outputStream):
		Mum.__init__(self, inputStream, textColumn, outputStream)
		self._transcripts = Transcripts()
		self.load(modelStream)
		self._errors = []
		self._stemmer = SwedishStemmer()
		self._filter = 'standard'
	
	def load(self, modelStream):
		"""
		Load model to use.
		"""
		data = json.loads(modelStream.read())
		self._model.setObjectData(data['model'])
		self._transcripts.setObjectData(data['transcripts'])
		return self
	
	def stemTokens(self, text):
		o = []
		for w in nltk.word_tokenize(text):
			if not w in SKIPCHARS: o += [self._stemmer.stem(w.lower())]
		return o
	
	def setInputFilter(self, name):
		self._filter = name
		return self
	
	def filterAndSplitText(self, text, replaceNums=True):
		for fltr in INPUT_FILTERS[self._filter]: text = fltr(text)
		if replaceNums: text = replaceNumbers(text)
		return self.stemTokens(text)
	
		
class NNTranslator(Translator):

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

class MarkovishTranslator(Translator):
	
	def __init__(self, modelStream, inputStream, textColumn, outputStream):
		self._model = Markovish()
		Translator.__init__(self, modelStream, inputStream, textColumn, False)
		self._outputStream = outputStream
		self._annotations = False
	
	def annotations(self, titrations):
		self._annotations = {'titrations':titration}
		return self
	
	def transcribe(self):
		#txt = '1 tablett varannan dag den första veckan, därefter 1 tablett dagligen'
		#txt = '25 mg dagligen i en vecka därefter 50 mg dagligen  mot tvångssyndrom'
		#txt = 'ENLIGT ORDINATION TABLETTER PÅ MORGONEN MOT ÅNGEST.'
		#tss = self._model.predictSequence(self.filterAndSplitText(txt))
		#print(self.filterAndSplitText(txt,False))
		#print(tss)
		#tss = self._transcripts.decode(' '.join(self.filterAndSplitText(txt,False)),' '.join(tss))
		#for x in tss:
		#	print(x.total())
		#print(tss)
		#print(tss[0].total(),tss[0].timePoint(),tss[0]._transcript,tss[1].total())
		#quit()
		N = 1
		correct = 0
		incorrect = ''
		#a = True
		#ts = []
		outputHeader = self.getHeaders() + ['recordID','transcriptID','mainTranscript','transcript','time','dose','object','error','novelObjects']
		output = '"'+'","'.join(outputHeader)+'"\n'
		while next(self):
			currentInput = self.getInput()
			#transcript = self._model.predictSequence(self.filterAndSplitText(self.getCurrentInput()))
			#transcript = ' '.join(transcript)
			novelObjects = 0
			tid = 1
			try:
				#transcripts = self._transcripts.decode(' '.join(self.filterAndSplitText(self.getCurrentInput(),False)),transcript)
				transcript, transcripts = self.getTranscript(self.getCurrentInput())
				for x in transcripts:
					tr = ' '.join(y[0] for y in x)
					#output += self.get('LOPNR')+',"'+self.getCurrentInput()+'",'+str(N)+','+str(tid)+',' + tr+','+str(x.timePoint())+','+str(x.total())+',"'+x.objects()+'"'+"\n"
					try:
						output += '"'+'","'.join(currentInput)+'",'+str(N)+','+str(tid)+','+transcript+',' + tr+','+str(x.timePoint())+','+str(x.total())+',"'+x.objects()+'",0,'+str(x.getNovelObjects())+"\n"
					#except TypeError as err2:
					except BaseException as err2:
						print('Error: ',err2)
						print('Current input:', currentInput)
						print('Timepoint: ',x.timePoint())
						print('Total: ',x.total())
						print('Objects: ',x.objects())
						print('Novel objects: ', x.getNovelObjects())
						quit()
					#print('Dose:',x.total())
					tid += 1
				correct += 1
			except BaseException as err:
				print('Error at '+str(N))
				print('\tInput: 	'+self.getCurrentInput())
				print('\tTranscript: '+transcript)
				print(err)
				#print(transcript)
				#print(self.filterAndSplitText(self.getCurrentInput()))
				incorrect += str(N)+': '+' '.join(self.filterAndSplitText(self.getCurrentInput(),False))+'|'+transcript+"\n"
				output += '"'+'","'.join(currentInput)+'",'+str(N)+','+str(tid)+',' + transcript+',,,,,1,'+"\n"
				#output += self.get('LOPNR')+',"'+self.getCurrentInput()+'",'+str(N)+','+str(tid)+',' + transcript+',,,1'+"\n"
			#self.setOutput()
			if transcript == '':
				output +=  '"'+'","'.join(currentInput)+'",'+str(N)+','+str(tid)+',' + transcript+',,,,,1,'+"\n"
			N += 1
		#print(incorrect)
		print(100*correct/N)
		#return ','.join(outputHeader) + "\n" + output
		return self._outputStream.write(output)
	
	def getTranscript(self, txt) -> list:
		if txt.strip() != '':
			transcript = self._model.predictSequence(self.filterAndSplitText(txt))
			transcript = ' '.join(transcript)
			transcripts = self._transcripts.decode(' '.join(self.filterAndSplitText(txt,False)),transcript)
			return transcript, transcripts
		else: return '',[]

class OperatorMarkovishTranslator(Translator):
	
	def __init__(self, modelStream, inputStream, textColumn, outputStream):
		self._model = OperatorMarkovish()
		Translator.__init__(self, modelStream, inputStream, textColumn, outputStream)
	
	def appendOperators(self, text, transcript):
		return self._model.predictSequence(self.filterAndSplitText(text), transcript)
	
