from .fileio import *
from fun.strman import rmExtraSpace
from fun.num import isNumeric
from .transcript import *
SKIPCHARS = ')(.?*=-+"!,\n'
class Transcripts:
	"""
	All text and patterns used as input is assumed to have been filtered and is clean.
	(1) Add text with accompanying transcript.
	(2) Store word<->qfot pairs
	(3) Create matrixes
	"""
	def __init__(self):
		# Words to numbers, e.g. five=5
		self._wordToNumeric = {}
		# Word to assigned value, eg. tablet or pill=pill
		self._wordToVal = {}
		self._wordToTime = {}
		self._ignoreChar = '/'
		self._numberChar = '#'
		self._funs = {'q':self.addWordToNumeric,'f':self.addWordToNumeric,'o':self.addWordToVal,'t':self.addTimeObject,'s':self.skip,'/':self.skip, ';':self.split,'*':self.setOperator}
		self._timeValues = {'d':1,'w':7}
		self._skips = 0
	
	def getObjectData(self):
		return {
			'wordToNumeric':self._wordToNumeric,
			'wordToVal':self._wordToVal,
			'wordToTime':self._wordToTime,
			'ignoreChar':self._ignoreChar,
			'numberChar':self._numberChar,
			'timeValues':self._timeValues
			}
	
	def setObjectData(self, data):
		self._wordToNumeric = data['wordToNumeric']
		self._wordToVal = data['wordToVal']
		self._wordToTime = data['wordToTime']
		self._ignoreChar = data['ignoreChar']
		self._numberChar = data['numberChar']
		self._timeValues = data['timeValues']
		return self

	def encode(self, text, pattern):
		"""
		Encode a text with a corresponding pattern.
		"""
		i = 0
		#txtParts = text.split(' ')
		txtParts = text
		pattern = rmExtraSpace(pattern.lower().strip())
		if pattern == '': pattern = 's'+str(len(txtParts))
		else: pattern = expandSkips(pattern)
		parts = pattern.split(' ')
		# Add up word codings
		for part in parts:
			if len(part) > 1 and part[1] == 'w' and text[i] == 'sedan':
				print(text)
				quit()
			if not part[0] in self._funs: raise KeyError(part[0] + ' is an unrecognized operator!')
			if not part in '*;':
				self._funs[part[0]](txtParts[i], part[1:])
				i += 1
		return self
	
	def decode(self, text, pattern, brute=True):
		"""
		Decode a text using a transcript.
		"""
		self._transcriptN = 0
		transcripts = []
		patterns = expandSkips(pattern).lower().split(';')
		if brute == True:
			patterns = patterns[0].split(' ')
			new = []
			newPat = []
			t = 0
			for x in patterns:
				if x == 't': t += 1
				new += [x]
				if t == 2:
					t = 0
					newPat += [' '.join(new)]
					new = []
			if len(new) != 0: newPat += [' '.join(new)]
			patterns = newPat
		i = 0
		txtParts = text.split(' ')
		for transcript in patterns:
			novelObjects = 0
			transcript = transcript.strip()
			#print(transcript)
			codings = []
			for part in transcript.split(' '):
				if part in 'qft':
					if isNumeric(txtParts[i]): codings += [(part,float(txtParts[i]))]
					elif part in 'qf': codings += [(part,self.wordToNumeric(txtParts[i]))]
					elif part == 't': codings += [(part,self.wordToTime(txtParts[i]))]
				elif part == 'o':
					obj = self.wordToVal(txtParts[i])
					if obj == False:
						obj = txtParts[i]
						novelObjects += 1
					codings += [(part,obj)]
				elif part == '*': codings += [('*',None)]
				if part != '*': i += 1
				#if not part in '*;': i += 1
			self._transcriptN += 1
			if len(codings) > 0:
				tra = Transcript2(codings)
				tra.setNovelObjects(novelObjects)
				transcripts += [tra]
		return transcripts
	
	def skip(self, word, operator):
		"""
		if n:
			if not isNumeric(n): raise ValueError('Character ' + word + ' coded as numeric ' + n + ' does not translate to numeric!')
			self._skips = int(n)
		self._skips -= 1
		return self._skips
		"""
		return word
	
	def split(self, pattern):
		"""
		Splits text and transcripts if there are multiple
		"""
		transcripts = []
		transcript = ''
		#txtParts = text.split(' ')
		i = 0
		for part in pattern.split(' '):
			if part == ';':
				transcripts += [transcript.rstrip()]
				transcript = ''
			elif part != '*':
				transcript += txtParts[i] + ' '
				i += 1
		if transcript != '': transcripts += [transcript]
		return transcripts
				
	
	def setOperator(self, word, operator):
		pass
	
	def wordToNumeric(self, word):
		if not word in self._wordToNumeric: raise ValueError('Word '+word+' does not have a numeric translation!')
		return self._wordToNumeric[word]
	
	def addWordToNumeric(self, word, number):
		"""
		For (q)uantity/(f)requency types. As all numbers are internally coded as #(.#)
		there's no need to add the numeric.
		"""
		if number:
			if not isNumeric(number): raise ValueError('Word ' + word + ' coded as numeric ' + number + ' does not translate to numeric!')
			self._wordToNumeric[word] = float(number)
	
	def wordToVal(self, word):
		return self._wordToVal[word] if word in self._wordToVal else False
	
	def addWordToVal(self, word, val):
		"""
		For (o)bject types.
		"""
		if val == '': raise ValueError(word + ' was assigned an empty value!')
		if word in self._wordToVal and self._wordToVal[word] != val: raise Exception(word + '=' + val + ' replaced ' + self._wordToVal[word])
		self._wordToVal[word] = val
	
	def wordToTime(self, word):
		if word in self._wordToTime: return self._wordToTime[word]
		else: return self.wordToNumeric(word)
	
	def addTimeObject(self, word, val):
		"""
		For (t)ime types.
		"""
		if val:
			if isNumeric(val): self.addWordToNumeric(word, val)#self._wordToNumeric[word] = val
			else: self._wordToTime[word] = val
	
"""	
	def __iter__(self):
		self._position = 0
		return self

	def __next__(self):
		self._position += 1
		if self._position > self.n: raise StopIteration
		return Transcript
    
	def rewind(self):
		self.position = 0
		return self
"""

"""
if __name__ == '__main__':
	t = Transcripts()
	s = Sequences(6,4, False)
	#t.addTrancript('one pill per day week 1 2 pills per day weeks two','q1 opill s2 tw t q opill s2 tw t2')
	#t._funs['q']('two','2')
	dire = '../../csv/'
	f = FileIO(open(dire+'dispensations_training_sample.csv'))
	output = 'id,time,dose\n'
	i = 0;
	while next(f) != False:
		print('Record: '+str(i)+' '+f.get('pattern'))
		t.addTrancript(f.get('doser'), f.get('pattern'))
		if f.get('pattern') != '':
			pattern = stripArguments(f.get('pattern').lower().replace(' ;','').replace(' *',''))
			s.add(f.get('doser'), pattern)
			for x in t.getTranscripts(f.get('doser'),stripArguments(f.get('pattern').lower())):
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
	s = Sequences()
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
	for word,n in s.getWordCount().items(): wrds += word +': ' + str(n) + "\n"
	#open(dire+'wordcount.txt','w').write(wrds)
	#print(s.getWordCount())
	#print(s.predictSequence('1 tablett till kvällen för sömn och minskad oro'))
	#print(t._wordToNumeric)
	#print(t._wordToTime)
"""
"""
one pill per day
q o s s
q = o s one pill
o = s s pill per
s = s per day

Prediction
OP1 = one pill
OP2 = OP1 pill per
OP3 = OP1 OP2 per day
OP4 = OP2 OP3 day
"""
