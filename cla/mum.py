from .fileio import FileIO
from .classifier import Classifier
from .transcripts import Transcripts
from fun.str import *
class Mum(FileIO, Classifier, Transcripts):
	"""
	Draft of parent class of all final classes. This class is to handle
	reading from a file with a column of text that is either being used
	for training or for prediction.
	"""
	def __init__(self, inputStream, textColumn, outputStream=False):
		"""
		There's always a file to read from.
		"""
		#self._replacor = re.compile('[0-9]')
		#self._replaceChar = '#'
		self.models = []
		self._textColumn = textColumn
		if inputStream: FileIO.__init__(self, inputStream, outputStream)
		Transcripts.__init__(self)
		Classifier.__init__(self)

	def transcribeCurrent(self):
		"""
		Returns (obj Translation, pattern, #error, #prob) of current record.
		"""
		txt = simplify(self.get(self._textColumn))
		txtNoNum = replaceNumbers(txt)
		pattern, p = self.classify(txtNoNum)
		#error = 0
		#val = False
		try:
			val = self.getTranslation(txt, pattern)
			error = 0
		except BaseException as e:
			val = False
			error = 1
		return val, pattern, error, p
