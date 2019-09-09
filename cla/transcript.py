from fun.num import isNumeric
class Transcript:
	def __init__(self, wordList):
		"""
		Instantiate by passing a list of tuples containing at least [0:('q',#), 1:('f',#)] but may contain f and o
		"""
		self.wordList = wordList
		self.position = 0
		self.n = len(self.wordList)
	
	def __iter__(self):
		self.position = 0
		return self
	 
	def __next__(self):
		self.position += 1
		if self.position > self.n: raise StopIteration
		return self.wordList[self.position-1]
	
	def rewind(self):
		self.position = 0
		return self
	
	def totals(self):
		"""
		Calculate total weight at each position (q or q*f)
		"""
		s = {}
		valPos = -1
		for p,x in enumerate(self):
			key, val = x
			if key == 'q':
				s[p] = val
				valPos = p
			elif key == 'f' and valPos > -1:
				s[valPos] = s[valPos] * val
		return s.values()
	
	def objects(self):
		s = []
		for p,x in enumerate(self):
			key, val = x
			if key == 'o':
				s += [val]
		return ','.join(s)
	
	def calculation(self):
		"""
		Return a string representation of the calculation performed
		"""
		s = ''
		for p,x in enumerate(self):
			key, val = x

def expandSkips(transcript):
	"""
	Transforms transcript of type "q f s2" opill to "q f s s opill"
	"""
	newTranscript = ''
	for part in transcript.split(' '):
		if part[0] == 's' and len(part) > 1 and isNumeric(part[1:]):
			nSkips = int(part[1:])
			while nSkips > 0:
				 newTranscript += 's '
				 nSkips -= 1
		else: newTranscript += part + ' '
	return newTranscript.rstrip()
	
def stripArguments(transcript):
	"""
	Remove arguments from operators.
	"""
	newTranscript = ''
	transcript = expandSkips(transcript)
	for part in transcript.split(' '): newTranscript += part[0] + ' '
	return newTranscript.rstrip()

class Transcript2:
	"""
	This class handles operations on transcripts.
	"""
	def __init__(self, transcript=False):
		if transcript != False: self.setTranscript(transcript)
		self._novelObjects = False
		self._timeUnitValues = {'d':1, 'w':7}
	
	def __iter__(self):
		self.position = 0
		return self
	 
	def __next__(self):
		self.position += 1
		if self.position > self.n: raise StopIteration
		return self._transcript[self.position-1]
	
	def rewind(self):
		self.position = 0
		return self
	
	def setNovelObjects(self,n):
		self._novelObjects = n
		return self
	
	def getNovelObjects(self):
		return self._novelObjects
	
	def setTranscript(self, transcript):
		"""
		transcript is a list of tuples containing at least [0:('q',#), 1:('f',#)] but may contain f and o
		"""
		#print(transcript)
		self._total = ''
		self._time = 0
		self._timeUnit = 'd'
		self._objects = []
		self._transcript = []
		# Remove skips
		for part in transcript:
			if part[0] != 's': self._transcript += [part]
		self.position = 0
		self.n = len(self._transcript)
		self._parse()
		return self

	def _parse(self):
		transcript = []
		# First extract objects and timepoints
		for operator, argument in self._transcript:
			if operator == 'o': self._objects += [argument]
			elif operator == 't':
				if isNumeric(argument): self._time = argument
				else: self._timeUnit = argument
			else: transcript += [(operator,argument)]
		# Solve multiplications first
		i = 0
		N = len(transcript)
		mafs = ''
		while i < N:
			operator = transcript[i][0]
			if i < N-1: nextOperator = transcript[i+1][0]
			else: nextOperator = ''
			if operator == '*':
				mafs += str(transcript[i-1][1]) + operator + str(transcript[i+1][1])
				#transcript[i-1] = ('q',transcript[i-1][1]*self._transcript[i+1][1])
				i += 1
			elif (operator == 'q' and nextOperator == 'f') or (operator == 'f' and nextOperator == 'q'):
				mafs += str(transcript[i][1]) +'*'+str(transcript[i+1][1])+'+'
				#transcript += [('q',transcript[i][1]*transcript[i+1][1])]
				i += 1
			else: mafs += str(transcript[i][1]) + '+'
			i += 1
		self._total = mafs.rstrip('+')
		#quit()
		"""
		# Now sum, extract objects and timepoints
		i = 0
		n = len(transcript)
		for operator, argument in transcript:
			#print(operator,argument)
			if operator == 'q': self._total += argument
			elif operator == 'o': self._objects += [argument]
			elif operator == 't':
				if isNumeric(argument): self._time = argument
				else: self._timeUnit = argument
		"""
		return transcript
	
	def total(self):
		"""
		Calculate total weight at each position (q or q*f)
		"""
		if self._total != '': return eval(self._total)
		else: return ''
	
	def objects(self,sep=','):
		return sep.join(self._objects)
	
	def time(self):
		return self._time
	
	def timeUnit(self):
		return self._timeUnit
	
	def timePoint(self):
		return self._timeUnitValues[self._timeUnit]*self._time
	
	def calculation(self):
		"""
		Return a string representation of the calculation performed
		"""
		s = ''
		for p,x in enumerate(self):
			key, val = x
