import numpy as np
import nltk
from nltk.stem.snowball import SwedishStemmer
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
        self.epochs = 20000
        self.dropOut = False
        self.dropOutPercent = 0.5
        # Layers
        self.synapse = False
        # Error tolerance in prediction
        self.error = 0.2
        self.nIncluded = 0
        self.setTrainRunLimit()
    
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
    
    def logit(self,x):
        return 1/(1 + np.exp(-x))

    def dLogitOutput(self,x):
        return x*(1 - x)
    
    def setNrHiddenNeurons(self, n=10):
        self.hiddenNeurons = n
        return self
    
    def updateNetwork(self,verbose=False):
        print('Neurons:',self.hiddenNeurons,'Patterns:',self.nIncluded)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        if verbose:
            print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (self.hiddenNeurons, str(self.alpha), self.dropOut, self.dropOutPercent if self.dropOut else '') )
            print ("Input matrix: %sx%s    Output matrix: %sx%s" % (len(self.x),len(self.x[0]),1, len(self.patterns)) )
        self.synapse = [0]*2
        last_mean_L2E = 1
        # Set random weights with mean 0. Argument is (N rows, N columns)
        # These are by default from a uniform distribution [0, 1). 2*[0,1)-1 gives [-1, 1)
        # Weight matrix Wx
        self.synapse[0] = 2*np.random.random((len(self.x[0]), self.hiddenNeurons)) - 1
        # Weight matrix Wy
        self.synapse[1] = 2*np.random.random((self.hiddenNeurons, len(self.includedPatterns))) - 1
        
        # Run optimization
        for j in iter(range(self.epochs+1)):
            # Feed forward through layers 0, 1, and 2
            #L0 = self.x
            # L1 = logit(L0 * Wx)
            #L1 = self.logit(np.dot(L0, self.synapse[0]))
            L1 = self.logit(np.dot(self.x, self.synapse[0]))
            if(self.dropOut):
                L1 *= np.random.binomial([np.ones((len(self.x),self.hiddenNeurons))],1-self.dropOutPercent)[0] * (1.0/(1-self.dropOutPercent))
            # L2 = logit(L1 * Wy) = logit(logit(L0 * Wx) * Wy)
            L2 = self.logit(np.dot(L1, self.synapse[1]))
            # Calculate the error
            L2E = self.y - L2
            if (j % 5000) == 0 and j > 5000:
                mean_L2E = np.mean(np.abs(L2E))
                # if this 10k iteration's error is greater than the last iteration, break out
                if mean_L2E < last_mean_L2E:
                    if verbose: print ("Prediction error after "+str(j)+" iterations:" + str(mean_L2E) )
                    #last_mean_error = np.mean(np.abs(L2E))
                    last_mean_L2E = mean_L2E#np.mean(np.abs(L2E))
                else:
                    if verbose: print ("Reached limit:", mean_L2E, ">", last_mean_L2E )
                    break
                
            #(Y-^Y) * W'
            L2D = L2E * self.dLogitOutput(L2)

            L1E = L2D.dot(self.synapse[1].T)

            L1D = L1E * self.dLogitOutput(L1)
            # Weight updates
            S1WU = (L1.T.dot(L2D))
            S0WU = (L0.T.dot(L1D))
        
            self.synapse[1] += self.alpha * S1WU
            self.synapse[0] += self.alpha * S0WU
    
    def getSynapses(self):
        return self.synapse
    
    def getWords(self):
        return self.words
    
    def getPatterns(self):
        return self.patterns
    
    def guestimate(self,text):
        #print(text)
        #print(self.createInput(text))
        # Multiply with input signals with hidden layer
        x = self.logit(np.dot(self.createInput(text), self.synapse[0]))
        # ...and the result with the output layer
        x = self.logit(np.dot(x, self.synapse[1]))
        return x
    
    def classify(self,text):
        p = self.guestimate(text)
        #print(p)
        #input('wait')
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
