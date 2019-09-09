import numpy as np
import nltk, datetime
from nltk.stem.snowball import SwedishStemmer
from scipy.optimize import minimize
from fun.strman import *

FUN = {
'identity':lambda x: x,
'logistic': lambda x: 1/(1+np.exp(-x)),
'square':lambda x: x*x
}
DFUN = {
'identity':lambda x: 1,
'logistic':lambda x: x*(1-x),
'square':lambda x: 2*x
}
def logistic (x):
	return 1/(1+np.exp(-x))

# transX is transformation that should be applied to x
def dLogistic (x,transX='identity'):
	x = FUN[transX](x)
	return DFUN['logistic'](x)

def sumOfSquares (x):
	x = x*x
	return sum(sum((x)))

def predict(x, wx, wy):
	return logistic(np.dot(logistic(np.dot(x, wx)),wy))
	
def calculateError(y, x, wx, wy):
	return sumOfSquares(y-predict(x, wx, wy))

def calculateErrorOptim (x, out, inp, neurons):
	xlen = len(inp[0])
	wx = np.array(x[:xlen*neurons]).reshape(xlen,neurons)
	wy = np.array(x[xlen*neurons:]).reshape(neurons,len(out[0]))
	#ypred = logit(np.dot(logit(np.dot(inp,wx)),wy))
	return calculateError(out, inp, wx, wy)

def matsToList(a, b):
	"""
	Convert to numpy matrices to list.
	"""
	return a.flatten().tolist() + b.flatten().tolist()

def calculateGradient(x, out, inp, neurons):
	xlen = len(inp[0])
	wx = np.array(x[:xlen*neurons]).reshape(xlen,neurons)
	wy = np.array(x[xlen*neurons:]).reshape(neurons,len(out[0]))
	# Output error: d(Y-Y(Wy*g(x)))^2/dWy = 2*(Y-Y(Wy*g(x)))*Y'(Wy*g(x))*g(x)
	L1 = logistic(np.dot(inp, wx))
	L2 = logistic(np.dot(L1, wy))
	error = out - L2
	dL2 = np.dot(L1.T,-2*error*dLogistic(L2))
	# Input error: d(Y-Y(Wy*g(x)))^2/dWx = 2*(Y-Y(Wy*g(x)))*Y'(Wy*g(x))*Wy*g'(x)*x
	dL1 = np.dot(inp.T, np.dot(-2*error*dLogistic(L2),wy.T) * dLogistic(L1))
	return np.append(np.asarray(dL1.flatten()), np.asarray(dL2.flatten()))

def calcError (beta, y, x, dimX,dimY):
	beta = np.array(beta).reshape(dimX,dimY)
	p = np.dot(x,beta)
	s = y-logistic(p)
	s2 = s*s
	return sum(sum(s2))

def grad (beta, y, x, dimX,dimY):
	beta = np.array(beta).reshape(dimX,dimY)
	xBeta = np.dot(x,beta)
	lg = logistic(xBeta)
	error = y-lg
	dx = logistic(-xBeta)*logistic(xBeta)
	g = np.dot(x.T,-2*error*dx)
	return np.asarray(g.flatten())

def calculateGradient2(x, out, inp, neurons):
	xlen = len(inp[0])
	wx = np.array(x[:xlen*neurons]).reshape(xlen,neurons)
	wy = np.array(x[xlen*neurons:]).reshape(neurons,len(out[0]))
	# Output error: d(Y-Y(Wy*g(x)))^2/dWy = 2*(Y-Y(Wy*g(x)))*Y'(Wy*g(x))*g(x)
	L1 = logistic(np.dot(inp, wx))
	L2 = logistic(np.dot(L1, wy))
	error = out - L2
	dL2 = np.dot(L1.T,-2*error*dLogistic(L2))
	# Input error: d(Y-Y(Wy*g(x)))^2/dWx = 2*(Y-Y(Wy*g(x)))*Y'(Wy*g(x))*Wy*g'(x)*x
	dL1 = np.dot(inp.T, np.dot(-2*error*dLogistic(L2),wy.T) * dLogistic(L1))
	return np.append(np.asarray(dL1.flatten()), np.asarray(dL2.flatten()))


class Model:
	"""
	The Model class is the parent of the different models that implement different ways
	of treating creating input and output matrices along with predictions
	"""
	def __init__(self):
		self._fit = False
	
	def fit(self):
		pass
	
	def predict(self):
		pass
	
	def createIO(self):
		pass
	
	def save(self, outputStream):
		pass
		
	def load(self, inputStream):
		pass
	
	def getObjectData(self):
		#data = {}
		#data['synapses'] = {}
		if not isinstance(self._fit,np.ndarray): raise TypeError('Error in saving model, seems no model has been fitted.')
		#for i,s in enumerate(self.synapse): data['synapses'][i] = s.tolist()
		#data['words'] = self.words
		#data['patterns'] = self.patterns
		return {'parameters':self._fit.tolist(),'dimParameters':self._fit.shape}

class Classifier:
	def __init__(self):
		self.stemmer = SwedishStemmer()
		self.ignore = ['?']
		self.words = []
		self.classified = []
		# Unique patterns
		self.patterns = []
		# Unique texts
		self.texts = []
		# Output
		self.y = []
		# Input
		self.x = []
		self.patternCounts = {}
		# Network parameters
		self.hiddenNeurons = 20
		self.alpha = 1
		# Maxium number of iterations during optimization
		self.epochs = 30000
		self.dropOut = False
		self.dropOutPercent = 0.5
		# Layers
		self.synapse = False
		# Error tolerance in prediction
		self.error = 0.2
		self.nIncluded = 0
		self.setTrainRunLimit()
		self.errors = []
		self.minimum = False
		self.minError = False
		self.optFun = {'logistic':logistic}
		self.inform = {0:lambda x: '', 1:lambda x: x}
		self.verbose = False
	
	def isInitiated(self):
		return self.synapse != False
	
	def addPattern(self, pattern, text):
		"""
		Add manually classified patterns with accompanying text
		"""
		if text not in self.texts and (pattern not in self.patterns or self.patternCounts[pattern] < self.maxExamples):
			self.texts += [text]
			self.classified += [{'c':pattern,'s':text}]
			if pattern not in self.patternCounts: self.patternCounts[pattern] = 1
			else: self.patternCounts[pattern] += 1
			if pattern not in self.patterns: self.patterns += [pattern]
			for w in self.stemTokens(text):
				if w not in self.words: self.words += [w]
			self.nIncluded = 0
			for p in self.patterns:
				if self.patternCounts[p] >= self.minExamples: self.nIncluded += 1
		return self
	
	def setTrainRunLimit(self, maxExamples=20, minExamples=3, minPatterns=2):
		"""
		The training sample for each pattern will increase until maxExamples is reached.
		Network will not run until minPatterns patterns are available.
		"""
		self.maxExamples = maxExamples
		self.minExamples = minExamples
		self.minPatterns = minPatterns
		return self
	
	def stemTokens(self, text):
		return [self.stemmer.stem(w.lower()) for w in nltk.word_tokenize(text)]
	
	def updateIO(self):
		"""
		Update input and output channels prior to optimizing
		"""
		self.x = []
		self.y = []
		self.nIncluded = 0
		self.includedPatterns = []
		# Start by determining the number of patterns to include
		for c in self.classified:
			if self.patternCounts[c['c']] >= self.minExamples and c['c'] not in self.includedPatterns:
				self.includedPatterns += [c['c']]
		self.nIncluded = len(self.includedPatterns)
		# Create output (Y) and input (X) matrixes
		for c in self.classified:
			if c['c'] in self.includedPatterns:
				x = self.createInput(c['s'])
				self.x += [x]
				y = [0]*self.nIncluded
				y[self.includedPatterns.index(c['c'])] = 1
				self.y += [y]
		return self
		
	def canRun(self):
		return self.nIncluded >= self.minPatterns
	
	def createInput(self,text):
		"""
		Create an input based on the correspondence between self.words and text
		"""
		t = self.stemTokens(text)
		x = []
		for w in self.words:
			x.append(1) if w in t else x.append(0)
		return x
	
	def logistic(self,x):
		return 1/(1 + np.exp(-x))

	def dLogistic(self,x):
		return x*(1 - x) 
	
	def setNrHiddenNeurons(self, n=10):
		self.hiddenNeurons = n
		return self
		
	def createLayers(self):
		synapses = [0]*2
		# Input connection
		synapses[0] = 2*np.random.random((len(self.x[0]), self.hiddenNeurons)) - 1
		# Output connection
		synapses[1] = 2*np.random.random((self.hiddenNeurons, len(self.includedPatterns))) - 1
		return synapses
	
	def predict(self,x, wx, wy):
		return logistic(np.dot(logistic(np.dot(x, wx)),wy))
	
	def calculateError(self,y, x, wx, wy):
		return sumOfSquares(y-self.predict(x, wx, wy))
	
	def matsToList(self, a, b):
		"""
		Convert to numpy matrices to list.
		"""
		return a.flatten().tolist() + b.flatten().tolist()
	
	def listToMats(self, l):
		"""
		Convert list to numpy matrices where the first elements are weights
		in X (input) and the rest weights in Y (output).
		"""
		lenx = len(self.x[0])
		leny = len(self.y[0])
		wx = np.array(l[:lenx*self.hiddenNeurons]).reshape(lenx, self.hiddenNeurons)
		wy = np.array(l[lenx*self.hiddenNeurons:]).reshape(self.hiddenNeurons, leny)
		return wx, wy
	
	def fit(self, useGradient=False, verbose=False):
		if verbose: print('Starting model fitting using', self.hiddenNeurons,'neurons in hidden layer.')
		#print(len(self.y[0]))
		p1, p2 = self.createLayers()
		params = self.matsToList(p1, p2)
		self.x = np.array(self.x)
		self.y = np.array(self.y)
		gradient = None
		if useGradient: gradient = calculateGradient
		minParams = minimize(fun=calculateErrorOptim, x0=params, args=(self.y, self.x, self.hiddenNeurons), jac=gradient, method='L-BFGS-B',options={'maxfun':50000})
		if verbose: print('Convergence=%s after %s iterations.' % (str(minParams.success), str(minParams.nit)))
		self.synapse = [0]*2
		self.synapse[0],self.synapse[1] = self.listToMats(minParams.x)

	def writeErrors(self):
		errors = "\n".join(self.errors)
		open('errors.csv','w').write(errors)
	
	def getSynapses(self):
		return self.synapse
	
	def getWords(self):
		return self.words
	
	def getPatterns(self):
		return self.patterns
	
	def guestimate(self,text):
		# Multiply with input signals with hidden layer
		x = self.logistic(np.dot(self.createInput(text), self.synapse[0]))
		# ...and the result with the output layer
		x = self.logistic(np.dot(x, self.synapse[1]))
		return x
	
	def classify(self,text):
		p = self.guestimate(text)
		# Assign index in self.patterns and probability of index [index,prob]
		results = [[i,r] for i,r in enumerate(p) if r > self.error]
		#results = [[i,r] for i,r in enumerate(p)]
		# Sort so that results with highest probability appears first
		results.sort(key=lambda x: x[1], reverse=True)
		if len(results) > 0: return self.patterns[results[0][0]],results[0][1]
		else: return '', 0
	
	def getObjectData(self):
		data = {}
		data['synapses'] = {}
		if not isinstance(self.synapse,list): raise TypeError('Synapses must be a list, but is '+str(type(self.synapse))+'. Seems no model has been fitted.')
		for i,s in enumerate(self.synapse): data['synapses'][i] = s.tolist()
		data['words'] = self.words
		data['patterns'] = self.patterns
		return data
	
	def setObjectData(self, data):
		self.synapse = [False]*len(data['synapses'])
		for s in data['synapses']: self.synapse[int(s)] = np.array(data['synapses'][s])
		self.words = data['words']
		self.patterns = data['patterns']
		return self
	
	def toJSON(self):
		"""
		Return everything necessary to reinstate the object as a JSON string
		"""
		return json.dumps(self.getObjectData())

class Markovish(Model):
	"""
	Predict pattern sequences.
	"""
	def __init__(self,trailingWords=2, trailingOperators=2):
		Model.__init__(self)
		self._verbose = 0
		self._verboseFun = {1:lambda x: print(x),0:lambda x: x}
		self._sentences = []
		self._transcripts = []
		self._wordColPos = {}
		self._wordCount = {}
		self._operatorColPos = {'q':0,'o':1,'f':2,'t':3,'s':4}
		self._operatorPosCol = [*self._operatorColPos.keys()]
		self._skipWords = []
		self.nRows = 0
		self._trailingWords = trailingWords
		self._trailingOperators = trailingOperators
		self._xOperatorsNColumns = []
		self._xSkipWordsNColumns = []
		self._xWordsNColumns = []
		self._constant = True
		
	def add(self, sentence, transcript):
		"""
		transcript = transcript.replace('/','s').split(' ')
		parts = self.stemTokens(simplify(replaceNumbers(sentence)))
		if len(transcript) < len(parts): transcript += ['s']*(len(parts)-len(transcript))
		"""
		self._transcripts += [transcript]
		for x in sentence:
			if not x in self._wordCount:
				self._wordCount[x] = 1
			else: self._wordCount[x] += 1
		self._sentences += [sentence]
		self.nRows += len(sentence)-1
		return self
	
	def getWordCount(self,reverse=True):
		o = {}
		for col,n in sorted(self._wordCount.items(), reverse=reverse, key=lambda x: x[1]): o[col] = n
		return o
	
	def createInput(self):
		"""
		Create the input (X) and output (Y) matrices
		"""
		self.verbose('Creating matrices...')
		# Start by assigning columns to each word that has at least  n observations
		for word,n in self._wordCount.items():
			if n > 1: self._wordColPos[word] = len(self._wordColPos)
			else: self._skipWords += [word]
		# These lists hold the respective dependent matrices
		Xwords = []
		Xoperators = []
		Swords = []
		for i in range(0,self._trailingWords):
			Xwords += [np.zeros((1, len(self._wordColPos)))]
			Swords += [np.zeros((1, 1))]
		for i in range(0, self._trailingOperators): Xoperators += [np.zeros((1,len(self._operatorColPos)))]
		Y = np.zeros((1, len(self._operatorColPos)))
		i = 0
		row = 0
		for x in self._transcripts:
			xLen = len(x)
			self.verbose('Current sentence: '+ ' '.join(self._sentences[i]))
			self.verbose('Current transcript: '+ ' '.join(x))
			# Iterate over each word in the sentence
			for j in range(0, xLen):
				Y[row,self._operatorColPos[x[j]]] = 1
				self.verbose('Current operator: '+x[j])
				self.verbose('Current word: '+self._sentences[i][j])
				# Add operators that trail the current operator (trail=0 means no trailing operators)
				if j >= 1:
					for trail in range(1, self._trailingOperators + 1):
						readPoint = j-trail
						if readPoint >= 0:
							readPoint = xLen-self._trailingOperators
							self.verbose('Trailing operator '+str(trail)+' operator '+x[j-trail])
							Xoperators[trail-1][row,self._operatorColPos[x[j-trail]]] = 1
				# Add up words from the word at the position of the current operator to the numer of trailing words
				nWords = len(self._sentences[i])
				for trail in range(0, self._trailingWords):
					if j+trail < nWords:
						self.verbose('Trailing word '+str(trail)+': '+self._sentences[i][j+trail])
						if self._wordCount[self._sentences[i][j+trail]] > 1: Xwords[trail][row,self._wordColPos[self._sentences[i][j+trail]]] = 1
						elif self._sentences[i][j+trail] in self._skipWords: Swords[trail][row,0] = 1
				# Add a new row to the output matrix
				Y = np.concatenate((Y,np.zeros((1,len(self._operatorColPos)))),axis=0)
				# Add new rows to the input matrixes
				for k in range(0,self._trailingOperators): Xoperators[k] = np.concatenate((Xoperators[k],np.zeros((1,len(self._operatorColPos)))),axis=0)
				for k in range(0,self._trailingWords):
					Xwords[k] = np.concatenate((Xwords[k],np.zeros((1,len(self._wordColPos)))),axis=0)
					Swords[k] = np.concatenate((Swords[k],np.zeros((1,1))),axis=0)
				row += 1
			#if i > 1: quit()
			i += 1
		# Drop any empty columns in the output matrix
		ySum = np.sum(Y,0)
		colDel = []
		for i in range(0,len(self._operatorColPos)):
			if ySum[i] == 0: colDel += [i]
		if len(colDel) > 0: Y = np.delete(Y,colDel,axis=1)
		# Remove the empty rows in each matrix (these exist as the loop ends with adding a row, fixed later)
		Y = np.delete(Y,row,axis=0)
		for k in range(0,self._trailingOperators): Xoperators[k] = np.delete(Xoperators[k],row,axis=0)
		for k in range(0,self._trailingWords):
			Xwords[k] = np.delete(Xwords[k], row, axis=0)
			Swords[k] = np.delete(Swords[k], row, axis=0)
		# Drop words with less that 2 observations
		return Y,Xwords,Xoperators,Swords
	
	def verbose(self, txt):
		self._verboseFun[self._verbose](txt)
		
	def getColNames(self):
		a = [[i,self.xColPos[i]] for i in self.xColPos]
		a.sort(key=lambda x: x[1], reverse=False)
		a = [i[0] for i in a]
		return a
	
	def fit(self):
		y, Xs, Xo, Sw = self.createInput()
		# Words matrix
		#self._xOperatorsNColumns = []
		#self._xSkipWordsNColumns = []
		self._xWordsNColumns = [Xs[0].shape[1]]
		x = Xs[0]
		for i in range(1,len(Xs)):
			x = np.concatenate((x,Xs[i]),axis=1)
			self._xWordsNColumns += [Xs[i].shape[1]]
		for i in range(0,len(Xo)):
			x = np.concatenate((x,Xo[i]),axis=1)
			self._xOperatorsNColumns += [Xo[i].shape[1]]
		for i in range(0,len(Sw)):
			x = np.concatenate((x, Sw[i]),axis=1)
			self._xSkipWordsNColumns += [Sw[i].shape[1]]
		# Add a constant
		#print(xWordsNColumns)
		#quit()
		x = np.concatenate((x,1+np.zeros((x.shape[0],1))),axis=1)
		np.set_printoptions(threshold=np.nan)
		dimX = x.shape[1]
		dimY = y.shape[1]
		print('Initiating model fitting...')
		"""
		if self._nnet != False:
			sx = 2*np.random.random((dimX, self._nnet)) - 1
			sy = 2*np.random.random((self._nnet, dimY)) - 1
			params = self.matsToList(sx,sy)
			m = minimize(fun=calculateErrorOptim, x0=params, args=(y,x,self._nnet),jac=calculateGradient2,method='L-BFGS-B',options={'maxfun':50000})
			wx = np.array(m.x[:dimX*self._nnet]).reshape(dimX, self._nnet)
			wy = np.array(m.x[dimX*self._nnet:]).reshape(self._nnet, dimY)
			fit = {'wx':wx,'wy':wy}
		else:
		"""
		params = 2*np.random.random((dimX,dimY)) - 1
		m = minimize(fun=calcError, x0=params, args=(y, x, dimX, dimY), jac=grad, method='L-BFGS-B',options={'maxfun':50000})
		fit = m.x.reshape(dimX,dimY)
		print('Convergence=%s after %s iterations.' % (str(m.success), str(m.nit)))
		self._fit = fit
		nOperatorCols = len(self._operatorColPos)
		nWordCols = len(self._wordColPos)
		return self._fit
	
	def predictSequence(self, text):
		"""
		Prepare the input matrix based on text and create the output.
		"""
		#parts = self.stemTokens(simplify(replaceNumbers(text)))
		sequence = []
		Xw0 = np.zeros(len(self._wordColPos))
		Sw0 = np.zeros(1)
		Xo0 = np.zeros(len(self._operatorColPos))
		Xw, Xo, Sw = [], [], []
		wordPos = 0
		#nWords = len(parts)
		nWords = len(text)
		# The first case will only include the first n words
		nn = 0
		for j in range(0, self._trailingWords):
			nn += len(self._wordColPos)
			Xw += [np.copy(Xw0)]
			Sw += [np.copy(Sw0)]
			if j+1 <= nWords:
				if text[j] in self._wordColPos: Xw[j][self._wordColPos[text[j]]] = 1
				elif text[j] in self._skipWords: Sw[j][0] = 1
		for j in range(0, self._trailingOperators):
			Xo += [np.copy(Xo0)]
			nn += len(self._wordColPos)
		#print(nn)
		# Store the predicted operators in a list
		operatorTrail = []
		while wordPos < nWords:
			# Score the next word, always at the last matrix
			if nWords-(wordPos+1) >= self._trailingWords:
				if text[wordPos+self._trailingWords - 1] in self._wordColPos: Xw[self._trailingWords-1][self._wordColPos[text[wordPos+self._trailingWords - 1]]] = 1
				elif text[wordPos+self._trailingWords - 1] in self._skipWords: Sw[self._trailingWords-1][0] = 1
			# Add up the preceding operators
			lenOperatorTrail = len(operatorTrail)
			if len(sequence) > 0 and self._trailingOperators > 0:
				if lenOperatorTrail == self._trailingOperators:
					operatorTrail.pop()
					lenOperatorTrail -= 1
				if len(sequence) > lenOperatorTrail:
					operatorTrail = [sequence[len(sequence)-1]] + operatorTrail
					lenOperatorTrail += 1
				for pos in range(0,lenOperatorTrail):
					X = np.copy(Xo0)
					X[self._operatorColPos[operatorTrail[pos]]] = 1
					Xo[pos] = np.copy(X)
			# Add up all matrices for multiplication
			x = np.concatenate(Xw+Xo+Sw,axis=0)
			# Add the constant
			x = np.concatenate((x,1+np.zeros(1)),axis=0)
			preds = self.predict(x)
			sequence += [preds[0][0]]
			# Remove the first matrices and append new ones at the end
			del Xw[0]
			del Sw[0]
			Xw += [np.copy(Xw0)]
			Sw += [np.copy(Sw0)]
			wordPos += 1
		return sequence
			
	
	def predict(self, inputs):
		"""
		Take the vector of inputs and multiply with the parameters.
		Return a list of outputs and associated probabilities.
		"""
		# Multiply the inputs with the corresponding parameters
		#if self._nnet != False:
		#	pred = logistic(np.dot(logistic(np.dot(inputs, self._fit['parameters']['wx'])),self._fit['X']['wy']))
		#else:
		pred = logistic(np.dot(inputs, self._fit))
		# Create list of the predicted operators and sort descendingly by their probability
		preds = [[self._operatorPosCol[i],prob] for i,prob in enumerate(pred)]
		preds.sort(key=lambda x: x[1], reverse=True)
		return preds
	
	def matsToList(self, a, b):
		"""
		Convert to numpy matrices to list.
		"""
		return a.flatten().tolist() + b.flatten().tolist()
	
	def getObjectData(self):
		# Extract coefficients from fitted model
		data = Model.getObjectData(self)
		data['fun'] = 'logistic'
		data['wordColPos'] = self._wordColPos
		data['trailingWords'] = self._trailingWords
		data['trailingOperators'] = self._trailingOperators
		data['order'] = ['words','operators','skipWords','constant']
		data['xWordsNColumns'] = self._xWordsNColumns
		data['xOperatorsNColumns'] = self._xOperatorsNColumns
		data['xSkipWordsNColumns'] = self._xSkipWordsNColumns
		data['constant'] = self._constant
		return data

	def setObjectData(self, data):
		self._fit = np.array(data['parameters'])
		self._wordColPos = data['wordColPos']
		self._trailingWords = data['trailingWords']
		self._trailingOperators = data['trailingOperators']
		return self

class OperatorMarkovish(Markovish):
	
	def __init__(self,trailingWords=2, trailingOperators=3):
		Markovish.__init__(self, trailingWords, trailingOperators)
		self._operatorColPos = {'+':0,'*':1,';':2}
		self._operatorPosCol = [*self._operatorColPos.keys()]
		self._funColPos = {'q':0,'o':1,'f':2,'t':3,'s':4}
		self._funPosCol = [*self._operatorColPos.keys()]
		self._verbose = 0
	
	def createInput(self):
		"""
		Create the input (X) and output (Y) matrices
		"""
		self.verbose('Creating matrices...')
		# Start by assigning columns to each word that has at least  n observations
		for word,n in self._wordCount.items():
			if n > 1: self._wordColPos[word] = len(self._wordColPos)
			else: self._skipWords += [word]
		# These lists hold the respective dependent matrices
		XwordsLead = [np.zeros((1, len(self._wordColPos)))]*self._trailingWords
		XwordsTrail = [np.zeros((1, len(self._wordColPos)))]*self._trailingWords
		SwordsLead = [np.zeros((1, 1))]*self._trailingWords
		SwordsTrail = [np.zeros((1, 1))]*self._trailingWords
		XoperatorsLead = [np.zeros((1,len(self._funColPos)))]*self._trailingOperators
		XoperatorsTrail = [np.zeros((1,len(self._funColPos)))]*self._trailingOperators
		#Xwords = []
		#Xoperators = []
		#Swords = []
		#for i in range(0,self._trailingWords*2):
		#	Xwords += [np.zeros((1, len(self._wordColPos)))]
		#	Swords += [np.zeros((1, 1))]
		#for i in range(0, self._trailingOperators*2): Xoperators += [np.zeros((1,len(self._funColPos)))]
		Y = np.zeros((1, len(self._operatorColPos)))
		i = 0
		row = 0
		for x in self._transcripts:
			xLen = len(x)
			self.verbose('Current sentence: '+ ' '.join(self._sentences[i]))
			self.verbose('Current transcript: '+ ' '.join(x))
			# One pill 2 times per day for 5 days
			# q + o + f + s + s + s + s + t * td
			# Iterate over each space between words
			wordPos = 1
			for j in range(1, xLen-1, 2):
				Y[row,self._operatorColPos[x[j]]] = 1
				self.verbose('Current operator: '+x[j])
				self.verbose('Current word: '+self._sentences[i][wordPos])
				# Add operators before/after the current function (trail=0 means no trailing operators)
				for trail in range(1, self._trailingOperators+1,2):
					readPoint1 = j-trail
					readPoint2 = j+trail
					op1 = x[readPoint1]
					op2 = x[readPoint2]
					if readPoint1 >= 0:
						#readPoint = xLen-self._trailingOperators
						self.verbose('Leading operator '+str(trail)+' operator '+op1)
						XoperatorsLead[trail][row, self._funColPos[op1]] = 1
					if readPoint2 < xLen:
						self.verbose('Trailing operator '+str(trail)+' operator '+op2)
						XoperatorsTrail[trail][row, self._funColPos[op2]] = 1
				# Add up words from the word at the position of the current operator to the numer of trailing words
				nWords = len(self._sentences[i])
				for trail in range(0, self._trailingWords):
					readPoint1 = wordPos-trail-1
					readPoint2 = wordPos+trail
					if readPoint1 >= 0:
						leadWord = self._sentences[i][readPoint1]
						self.verbose('Leading word '+str(trail+1)+': '+leadWord)
						if self._wordCount[leadWord] > 1: XwordsLead[trail][row, self._wordColPos[leadWord]] = 1
						elif leadWord in self._skipWords: SwordsLead[trail][row,0] = 1					
					if readPoint2 < nWords:
						trailWord = self._sentences[i][readPoint2]
						self.verbose('Trailing word '+str(trail+1)+': '+trailWord)
						if self._wordCount[trailWord] > 1: XwordsTrail[trail][row, self._wordColPos[trailWord]] = 1
						elif trailWord in self._skipWords: SwordsTrail[trail][row,0] = 1
				# Add a new row to the output matrix
				Y = np.concatenate((Y,np.zeros((1,len(self._operatorColPos)))),axis=0)
				# Add new rows to the input matrixes
				for k in range(0,self._trailingOperators):
					XoperatorsLead[k] = np.concatenate((XoperatorsLead[k],np.zeros((1,len(self._funColPos)))),axis=0)
					XoperatorsTrail[k] = np.concatenate((XoperatorsTrail[k],np.zeros((1,len(self._funColPos)))),axis=0)
				for k in range(0,self._trailingWords):
					XwordsLead[k] = np.concatenate((XwordsLead[k],np.zeros((1,len(self._wordColPos)))),axis=0)
					XwordsTrail[k] = np.concatenate((XwordsTrail[k],np.zeros((1,len(self._wordColPos)))),axis=0)
					SwordsLead[k] = np.concatenate((SwordsLead[k],np.zeros((1,1))),axis=0)
					SwordsTrail[k] = np.concatenate((SwordsTrail[k],np.zeros((1,1))),axis=0)
				row += 1
				wordPos += 1
			#if i > 1: quit()
			i += 1
			#print(i)
		# Drop any empty columns in the output matrix
		ySum = np.sum(Y,0)
		#print(ySum)
		colDel = []
		for i in range(0,len(self._operatorColPos)):
			if ySum[i] == 0: colDel += [i]
		if len(colDel) > 0: Y = np.delete(Y,colDel,axis=1)
		# Remove the empty rows in each matrix (these exist as the loop ends with adding a row, fixed later)
		Y = np.delete(Y,row,axis=0)
		for k in range(0,self._trailingOperators):
			XoperatorsLead[k] = np.delete(XoperatorsLead[k],row,axis=0)
			XoperatorsTrail[k] = np.delete(XoperatorsTrail[k],row,axis=0)
		for k in range(0,self._trailingWords):
			XwordsLead[k] = np.delete(XwordsLead[k], row, axis=0)
			XwordsTrail[k] = np.delete(XwordsTrail[k], row, axis=0)
			SwordsLead[k] = np.delete(SwordsLead[k], row, axis=0)
			SwordsTrail[k] = np.delete(SwordsTrail[k], row, axis=0)
		Xwords = XwordsLead+XwordsTrail
		Xoperators = XoperatorsLead+XoperatorsTrail
		Swords = SwordsLead+SwordsTrail
		# Drop words with less that 2 observations
		return Y,Xwords,Xoperators,Swords

	def predictSequence(self, text, transcript):
		"""
		Prepare the input matrix based on text and create the output.
		"""
		# Remove trailing skips and the corresponding words
		#newText, newTranscript = [], []
		#for i, fn in enumerate(transcript):
		"""
		newSentence = []
		newTranscript = []
		for i, wrd in enumerate(sentence):
			if transcript[i] != 's':
				newSentence += [transcript[i]]
				newTranscript += [transcript[i]]
		text = newSentence
		transcript = newTranscript
		"""
		#parts = self.stemTokens(simplify(replaceNumbers(text)))
		sequence = []
		Xw0 = np.zeros(len(self._wordColPos))
		Sw0 = np.zeros(1)
		Xo0 = np.zeros(len(self._funColPos))
		wordPos = 0
		funPos = 0
		lenTranscript = len(transcript)
		nWords = len(text)
		# The first case will only include the first n words
		#XwLead,XwTrail, XoLead,XoTrail, SwLead,SwTrail = [], [], [], [], [], []
		#XwLead = [np.copy(Xw0)]*self._trailingWords
		#XwTrail = [np.copy(Xw0)]*self._trailingWords
		#SwLead = [np.copy(Sw0)]*self._trailingWords
		#SwTrail = [np.copy(Sw0)]*self._trailingWords
		#XoLead = [np.copy(Xo0)]*self._trailingOperators
		#XoTrail = [np.copy(Xo0)]*self._trailingOperators
		#for j in range(0, self._trailingWords):
		#	XwLead += [np.copy(Xw0)]
		#	SwLead += [np.copy(Sw0)]
		#	XwTrail += [np.copy(Xw0)]
		#	SwTrail += [np.copy(Sw0)]
		#for j in range(0, self._trailingOperators):
		#	XoLead += [np.copy(Xo0)]
		#	XoTrail += [np.copy(Xo0)]
		# Store the predicted operators in a list
		#0 1 2
		#q o f
		#print(lenTranscript,transcript,text)
		operatorTrail = []
		for op in range(1, lenTranscript):
			sequence += [transcript[op-1]]
			XwLead = [np.copy(Xw0)]*self._trailingWords
			XwTrail = [np.copy(Xw0)]*self._trailingWords
			SwLead = [np.copy(Sw0)]*self._trailingWords
			SwTrail = [np.copy(Sw0)]*self._trailingWords
			XoLead = [np.copy(Xo0)]*self._trailingOperators
			XoTrail = [np.copy(Xo0)]*self._trailingOperators
		#while wordPos < nWords:
			# Score the next word, always at the last matrix
			for trail in range(0, self._trailingWords):
				if op-trail-1 >= 0:
					wrd = text[op-trail-1]
					if wrd in self._wordColPos: XwLead[trail][self._wordColPos[wrd]] = 1
					else: SwLead[trail][0] = 1
				if op+trail < nWords:
					wrd = text[op+trail]
					if wrd in self._wordColPos: XwTrail[trail][self._wordColPos[wrd]] = 1
					else: SwTrail[trail][0] = 1
			for trail in range(0, self._trailingOperators):
				if op-trail-1 <= 0: XoLead[trail][self._funColPos[transcript[op-trail-1]]] = 1
				if op+trail < nWords: XoTrail[trail][self._funColPos[transcript[op+trail]]] = 1
			"""
			if nWords-(wordPos+1) >= self._trailingWords:
				if text[wordPos+self._trailingWords - 1] in self._wordColPos: Xw[self._trailingWords-1][self._wordColPos[text[wordPos+self._trailingWords - 1]]] = 1
				elif text[wordPos+self._trailingWords - 1] in self._skipWords: Sw[self._trailingWords-1][0] = 1
			# Add up the preceding operators
			lenOperatorTrail = len(operatorTrail)
			if len(sequence) > 0 and self._trailingOperators > 0:
				if lenOperatorTrail == self._trailingOperators:
					operatorTrail.pop()
					lenOperatorTrail -= 1
				if len(sequence) > lenOperatorTrail:
					operatorTrail = [sequence[len(sequence)-1]] + operatorTrail
					lenOperatorTrail += 1
				for pos in range(0,lenOperatorTrail):
					X = np.copy(Xo0)
					X[self._operatorColPos[operatorTrail[pos]]] = 1
					Xo[pos] = np.copy(X)
			"""
			# Add up all matrices for multiplication
			x = np.concatenate(XwLead+XwTrail+XoLead+XoTrail+SwLead+SwTrail,axis=0)
			# Add the constant
			x = np.concatenate((x,1+np.zeros(1)),axis=0)
			preds = self.predict(x)
			if preds[0][0] != '+': sequence += [preds[0][0]]
			# Remove the first matrices and append new ones at the end
			#del Xw[0]
			#del Sw[0]
			#Xw += [np.copy(Xw0)]
			#Sw += [np.copy(Sw0)]
			wordPos += 1
		# Add up the last operator
		sequence += [transcript[op]]
		return sequence

	def getObjectData(self):
		# Extract coefficients from fitted model
		data = Model.getObjectData(self)
		data['fun'] = 'logistic'
		data['wordColPos'] = self._wordColPos
		data['trailingWords'] = self._trailingWords
		data['trailingOperators'] = self._trailingOperators
		data['order'] = ['words','operators','skipWords','constant']
		data['xWordsNColumns'] = self._xWordsNColumns
		data['xOperatorsNColumns'] = self._xOperatorsNColumns
		data['xSkipWordsNColumns'] = self._xSkipWordsNColumns
		data['constant'] = self._constant
		return data

	def setObjectData(self, data):
		self._fit = np.array(data['parameters'])
		self._wordColPos = data['wordColPos']
		self._trailingWords = data['trailingWords']
		self._trailingOperators = data['trailingOperators']
		return self
