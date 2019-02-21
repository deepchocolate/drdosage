#!/usr/bin/env python3
# coding=utf8
#from cla import *
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
<<<<<<< HEAD
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
		pass
>>>>>>> 5ef397374b7fded7518bf248ccf064af3c28915a
