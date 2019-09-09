import csv
class FileIO(csv.DictReader):
	def __init__(self, inStream, outStream=False):
		self._inFile = inStream
		self._outFile = outStream
		self._inputResource = inStream
		self._inputStream = False
		self._inputHeader = False
		self._outputHeader = False
		self._outputStream = False
		self._currentInput = False
		self._currentOutput = False
		self._nrRecords = False
		#self._errors = []
		self._output = False
		self._position = -1
		self.loadInput()
		#self.loadOutput()
		
	def __next__(self):
		#if self._position >= 0: self.setOutput()
		try:
			self._position += 1
			self._currentInput = csv.DictReader.__next__(self)#super(FileIO, self).__next__()
			self._currentOutput = self._currentInput
			return self._currentInput
		except StopIteration: return False
	
	def loadInput(self):
		"""
		Load data from input resource.
		"""
		#self._inputResource = open(inFile,newline='')
		#try: self._inputResource = open(self._inFile,newline='')
		#except IOError: print('File "'+inFile+'" could not be opened!')
		#print(resource)
		#self._nrRecords = self._inputResource.read().count("\n")-1
		self.seek()
		dialect = csv.Sniffer().sniff(self._inputResource.read(2048))
		self.seek()
		super(FileIO,self).__init__(self._inputResource,dialect=dialect)
		# Dirty but accurate records counting
		i = 0
		while next(self) != False: i += 1
		self._nrRecords = i
		self.rewind()
		self._output = [False]*self._nrRecords
		self._outputHeader = self.fieldnames
		return self
	
	def getHeaders(self):
		return self._outputHeader

	def loadOutput(self):
		"""
		Load final output.
		"""
		self._outputStream = csv.DictWriter(self._outFile, self._outputHeader)
		self._outputStream.writeheader()
		return self
		
	def hasOutput(self):
		if self._output[self._position] == False: return False
		else: return True

	def seek(self, bytePos=0):
		self._inputResource.seek(bytePos)
		return self

	def rewind(self):
		"""
		Reset file position to beginning.
		"""
		self.seek()
		# Skip header
		next(self)
		self._position = -1
		return self

	def get(self, column):
		"""
		Read column from input stream
		"""
		return self._currentInput[column]
	
	def getInput(self):
		return self._currentInput.values()

	def getOutput(self,position=False):
		"""
		Return output to be written at row position (defaults to current position).
		"""
		if position == False: position = self._position
		return self._output[position]

	def setOutput(self):
		"""
		Pass current output to final output data.
		"""
		# Position is always one ahead
		if self._position < self._nrRecords: self._output[self._position] = self._currentOutput
		return self
	
	def setCol(self, column, value):
		"""
		Set output column to value.
		"""
		if column not in self._outputHeader: self._outputHeader += [column]
		self._currentOutput[column] = value
		return self

	def writeOutput(self):
		if self._outputStream == False: self.loadOutput()
		#print(len(self._output))
		self._outputStream.writerows(self._output)
		self._output = []
	
	def nrRecords(self):
		return self._nrRecords
	
	def position(self):
		return self._position

	def posPct(self, digits=2):
		return round(100*self._position/self._nrRecords, 2)
