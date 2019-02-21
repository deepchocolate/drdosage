#!/usr/bin/env python3
# coding=utf8
import operator
#import nltk
# Run import nltk; nltk.download('punkt')
import os, sys, argparse, json, datetime, numpy as np, re, csv
#from myCsv import *
### Helper functions: isNumeric() and rmExtraSpace()
from fun import rmExtraSpace
# Classes
from cla import *

class Mum(FileIO):
	"""
	Draft of parent class of all final classes. This class is to handle
	reading from a file with a column of text that is either being used
	for training or for prediction.
	"""
	def __init__(self, inputStream, inputColumn):
		"""
		There's always a file to read from.
		"""
		super(FileIO,self).__init__(inputStream)
	

class Trainer(Mum):
	def __init__(self, trainStream, textColumn, patternColumn, modelFile, optStream=False, optColumn=False):
		"""
		trainStream: File with training data.
		textColumn: Column number or name of text.
		patternColumn: Column number or name of pattern.
		modelFile: File to save the model in.
		optStream: If optimizing using text in trainFile instead.
		optColumn: Column of text when using optFile.
		"""
		# If optStream is passed, then load training data separately.
		if optStream:
			inp = optStream
			self.trainFromFile(trainStream, textColumn, patternColumn)
		else: inp = trainStream
	
	def trainFromFile(self, fileName: str, textColumn=1, patternColumn=2):
        """
        Load training data from file. If textColumn and patternColumn are not provided
        it is assumed that text is in the first column and pattern in the second. These
        parameters may be provided as column name (string) or column number (integer).
        """
        self.classifier.setTrainRunLimit(30,6,2)
        res = open(fileName, newline='')
        dialect = csv.Sniffer().sniff(res.read(2048))
        res.seek(0)
        if type(patternColumn) == int and type(patternColumn) == int:
            fio = csv.reader(res, dialect=dialect)
            next(fio)
            textColumn -= 1
            patternColumn -= 1
        elif type(patternColumn) == str and type(patternColumn) == str:
            fio = csv.DictReader(res, dialect=dialect)
        i = 1
        included = 0
        for row in fio:
            txt = self.simplify(row[textColumn])
            txtNoNum = self.replacor.sub(self.replaceChar, txt)
            print(txtNoNum)
            if row[patternColumn] not in ['-','']:
                try:
                    transcript = self.transcripts.parseInstruction(txt, row[patternColumn].lower())
                    #val = self.transcripts.getTranslation(row[textColumn], transcript)
                    self.classifier.addPattern(transcript, txtNoNum)
                    included += 1
                except BaseException:
                    print('Unable to parse record',i,':',row[textColumn],row[patternColumn])
            i += 1
        self.classifier.updateIO()
        print('Parsed ',included,' records. Proceeding to optimize...')
        optNeurons, accuracy = self.optimize()
        print(round(100*accuracy,2),'% accuracy reached with',optNeurons,' neurons')
        self.classifier.setNrHiddenNeurons(optNeurons).updateNetwork3()
        return self

class Translator:

    minPatterns = 2
    maxExamples = 20
    minExamples = 10

    def __init__(self, inFile, outFile=False, column=False):
        # Use a complied regex to replace characters fed into classifier
        self.replacor = re.compile('[0-9]')
        self.replaceChar = '#'
        self.column = column
        # Counter
        self.incorrect = 0
        self.outFile = outFile
        self.instantiateModels()
        #self._io = FileIO(inFile, outFile)
        self.models = []
        self._ioData = FileIO(inFile, outFile)
        self._errors = [True]*self._ioData.nrRecords()
    
    def loadIO (self, inFile, outFile=False):
        """
        Load input/output files. Output is optional.
        """
        self._ioData = FileIO(inFile, outFile)
        return self
    
    def instantiateModels(self):
        self.transcripts = Transcripts()
        self.classifier = Classifier()
        return self
        
    def goToNextMissing(self):
        """
        Iterate until next untranscribed text
        """
        while self._ioData.position() < self._ioData.nrRecords() and self._errors[self._ioData.position()] == False: next(self._ioData)
    
    def optimize(self, minNeurons=8, maxNeurons=20):
        """
        Try out different number of hidden neurons and find the one yielding most correct
        """
        optimal = 0
        accuracy = 0
        maxAccuracy = 0
        # Training has been performed up to this position
        for neurons in range(minNeurons, maxNeurons):
            self._ioData.rewind()
            self.goToNextMissing()
            incorrect = 0
            self.classifier.setNrHiddenNeurons(neurons).updateNetwork3()
            i = 0
            while next(self._ioData) != False and i < 10000:
                if self._ioData.position() % 5000 == 0: print('Parsing record:',self._ioData.position() + 1)
                #txt = self.simplify(self._ioData.get(self.column))
                #txtNoNum = self.replacor.sub(self.replaceChar, txt)
                #pattern, p = self.classifier.classify(txtNoNum)
                #try:
                #    val = self.transcripts.getTranslation(txt, pattern)
                #    if val == False: incorrect += 1
                #except BaseException as e: incorrect += 1
                val, pattern, error, p = self.transcribeCurrent()
                if val == False: incorrect += 1
                i += 1
            #accuracy = 1 - incorrect/self.inFile.nrLines
            accuracy = 1
            if incorrect < i: accuracy = 1 - incorrect/i
            if accuracy > maxAccuracy:
                optimal = neurons
                maxAccuracy = accuracy
            print('% transcribed:',round(100*accuracy,2))
        return optimal, maxAccuracy
    
    def run(self):
        pos = self.train()
        # Now apply this procedure allowing the nr of neurons to vary to find the optimal for the given data
        print('Training target reached. Proceeding to optimize...')
        optNeurons, accuracy = self.optimize(8,20)
        print(round(100*accuracy,2),'% accuracy reached with',optNeurons,' neurons')
        self.classifier.setNrHiddenNeurons(optNeurons).updateNetwork()
        print('Classifying...')
        # pos is position of last record processed manually, start transcription at the next.
        self.transcribe()
    
    def train(self):
        while self.classifier.canRun() == False and next(self._ioData) != False:
            val = False
            pattern = ''
            print('Parsing record:',self._ioData.position() + 1)
            txt = self.simplify(self._ioData.get(self.column))
            txtNoNum = self.replacor.sub(self.replaceChar, txt)
            transcript = self.transcripts.getInstruction(txt)
            if transcript != False:
                transcript = transcript.rstrip('/')
                val = self.transcripts.getTranslation(txt, transcript)
                pattern = transcript
                self.classifier.addPattern(transcript, txtNoNum)
            total = ''
            objects = ''
            if val != False:
                total = sum(val.totals())
                objects = val.objects()
            self._ioData.setCol('total', total).setCol('manual', 1).setCol('pattern', pattern).setCol('error', 0).setCol('prob','').setCol('object', objects)
            self._errors[self._ioData.position()] = False
            self._ioData.setOutput()
        self.classifier.updateIO()
        return self._ioData.position()
    
    def trainFromFile(self, fileName: str, textColumn=1, patternColumn=2):
        """
        Load training data from file. If textColumn and patternColumn are not provided
        it is assumed that text is in the first column and pattern in the second. These
        parameters may be provided as column name (string) or column number (integer).
        """
        self.classifier.setTrainRunLimit(30,6,2)
        res = open(fileName, newline='')
        dialect = csv.Sniffer().sniff(res.read(2048))
        res.seek(0)
        if type(patternColumn) == int and type(patternColumn) == int:
            fio = csv.reader(res, dialect=dialect)
            next(fio)
            textColumn -= 1
            patternColumn -= 1
        elif type(patternColumn) == str and type(patternColumn) == str:
            fio = csv.DictReader(res, dialect=dialect)
        i = 1
        included = 0
        for row in fio:
            txt = self.simplify(row[textColumn])
            txtNoNum = self.replacor.sub(self.replaceChar, txt)
            print(txtNoNum)
            if row[patternColumn] not in ['-','']:
                try:
                    transcript = self.transcripts.parseInstruction(txt, row[patternColumn].lower())
                    #val = self.transcripts.getTranslation(row[textColumn], transcript)
                    self.classifier.addPattern(transcript, txtNoNum)
                    included += 1
                except BaseException:
                    print('Unable to parse record',i,':',row[textColumn],row[patternColumn])
            i += 1
        self.classifier.updateIO()
        print('Parsed ',included,' records. Proceeding to optimize...')
        optNeurons, accuracy = self.optimize()
        print(round(100*accuracy,2),'% accuracy reached with',optNeurons,' neurons')
        self.classifier.setNrHiddenNeurons(optNeurons).updateNetwork3()
        return self
    
    def transcribeCurrent(self):
        """
        Returns (obj Translation, pattern, #error, #prob) of current record.
        """
        txt = self.simplify(self._ioData.get(self.column))
        txtNoNum = self.replacor.sub(self.replaceChar, txt)
        pattern, p = self.classifier.classify(txtNoNum)
        #error = 0
        #val = False
        try:
            val = self.transcripts.getTranslation(txt, pattern)
            error = 0
        except BaseException as e:
            val = False
            error = 1
        return val, pattern, error, p
            
    def transcribe (self):
        self._ioData.rewind()
        self.goToNextMissing()
        incorrect = 0
        while next(self._ioData) != False:
            if self._ioData.position() % 5000 == 0: print('Parsing record:', self._ioData.position() + 1)
            #txt = self.simplify(self._ioData.get(self.column))
            #txtNoNum = self.replacor.sub(self.replaceChar, txt)
            #pattern, p = self.classifier.classify(txtNoNum)
            #error = 0
            #val = False
            #try:
            #    val = self.transcripts.getTranslation(txt, pattern)
            #except BaseException as e:
            #    val = False
            #    error = 1
            total = ''
            objects = ''
            val, pattern, error, p = self.transcribeCurrent()
            if val != False:
                total = sum(val.totals())
                objects = val.objects()
                self._errors[self._ioData.position()-1] = False
            else: error = 1
            self._ioData.setCol('total', total).setCol('manual', 0).setCol('pattern', pattern).setCol('error', error).setCol('prob', p).setCol('object', objects)
            self._ioData.setOutput()
        return incorrect
            
    def simplify(self, txt):
        words = rmExtraSpace(txt).lower().split(' ')
        o = ''
        for w in words: o += w.strip('.?*=-+"!,') + ' ' 
        return rmExtraSpace(o.strip())
    
    def saveModel(self):
        data = {}
        data['classifier'] = self.classifier.getObjectData()
        data['transcripts'] = self.transcripts.getObjectData()
        self.models += [data]
        return self
    
    def save(self, fileName):
        f = open(fileName, 'w',encoding='utf8')
        f.write(json.dumps(self.models))
        f.close()
    
    def load(self, fileName):
        self.models = json.loads(open(fileName).read())
        return self
    
    def runModels(self):
        inc = 0
        for i,m in enumerate(self.models):
            print('Load model', i)
            self.classifier.setObjectData(m['classifier'])
            self.transcripts.setObjectData(m['transcripts'])
            print('Run model', i)
            inc += self.transcribe()
            print('% transcribed:',round(100*(1-sum(self._errors)/self._ioData.nrRecords()),2))
        print('Done')
        self._ioData.writeOutput()

def menu():
    inp = '1'
    minPatterns = Translator.minPatterns
    maxExamples = Translator.maxExamples
    minExamples = Translator.minExamples
    while inp != '3':
        print('[1] Train\n[2] Set limits (# max examples, # min examples, # min patterns)\n[3] Write out and quit')
        inp = input('Choice:')
        if inp == '1':
            self.classifier.setTrainRunLimit(maxExamples, minExamples, minPatterns)
            self.run()
            self.saveModel().instantiateModels()
            self._ioData.rewind()
            errors = sum(self._errors)
            if errors > 0 and errors < self._ioData.nrRecords(): print('Total % transcribed:',round(100*(1-sum(self._errors)/self._ioData.nrRecords()),2))
            else:
                print('All done..exiting')
        elif inp != '3':
            maxExamples, minExamples, minPatterns = inp.split(' ')
        else:
            print('Writing output...')
            self._ioData.writeOutput()
            self.save('synapses.json')

if __name__ == '__main__':
    #print('hej')
    ap = argparse.ArgumentParser(description="""Dr. Dose is a neural network based prediction engine tailored to 
    extract dosage information from pharmaceutial prescription texts in the Swedish Prescribed Drug Register.
    
    If model is unspecified the default action is to train data and output. In/outfile needs to be in a CSV-format.""")
    ap.add_argument('-f', metavar='filter', nargs=1, help='Filter text. Needs to be done for training data.')
    ap.add_argument('-m', metavar='model', nargs=1, help='Model to load and apply to [input] data or file.')
    ap.add_argument('-t', metavar='train', nargs=1, help='Fit model to input.')
    ap.add_argument('-n', metavar='neurons', nargs=1, help='Set no. neurons in hidden layer. Only when training.')
    ap.add_argument('[input file]', type=argparse.FileType('r'), nargs='?', default=sys.stdin, help='File to read from, if omitted read from standard in.')
    ap.add_argument('[output file]', type=argparse.FileType('w'), nargs='?', default=sys.stdout, help='File to write to, if omitted output to standard out.')
    if len(sys.argv) > 1:
        args = ap.parse_args()
        pass
    else:
        #t = Translator(open('../csv/meds_50000.csv'),open('meds_50000_output.csv','w'),'doser')
        t = Translator(open('../csv/meds_100.csv'),open('meds_100_output.csv','w'),'doser')
        #print(t.trainFromFile('../csv/meds_100_train.csv','doser','pattern').saveModel().save('meds_200.model'))
        t.load('meds_200.model').runModels()
        #menu()
        
    #print('hej')
#else:
#    try:
#        print('d')
        #t = Translator('meds_100_train.csv','meds_100_output.csv','doser')#.menu()
        #t = Translator('meds_50000.csv','meds_100_output.csv','doser')#.menu()
        #print(t.trainFromFile('meds_100_train.csv','doser','pattern').saveModel().save('meds_200.model'))
        #t.load('meds_200.model').runModels()
#    except KeyboardInterrupt:
#        pass

# Lello et al BIORXIV deep learning
# Fix indicators for range, weekdays, sometimes
