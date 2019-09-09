#!/usr/bin/env python3
# coding=utf8
from cla.translator import *
import argparse, sys, json


if __name__ == '__main__':
	ap = argparse.ArgumentParser(description="""Dr. Dose is a prediction engine tailored to 
	extract dosage information from pharmaceutial prescription texts in the Swedish Prescribed Drug Register.
	
	If data is read from a file this needs to be in a CSV format. If data is passed through standard in input data
	must consist of text sequences separated by linebreaks. Output file be provided in a CSV format with the same
	structure as input, but additional columns of pattern, total, probability, and error.""")
	ap.add_argument('model', metavar='prediction model', nargs=1, type=argparse.FileType('r'), help='Model to load and apply to [input] data or file.')
	ap.add_argument('-i','--input-file', metavar='input data', type=argparse.FileType('r'), nargs='?', default=sys.stdin, help='File to read from, if omitted read from standard in.')
	ap.add_argument('-ic','--input-column', metavar='input text column', type=str, nargs=1, help='Input text column.',required=True)
	ap.add_argument('-o','--output-file', metavar='output data', type=argparse.FileType('w'), nargs='?', default=sys.stdout, help='File to write to, if omitted output to standard out.')
	ap.add_argument('-a','--annotations', metavar='annotate', type=list, nargs='?', default=[], help='Annotations to make.')
	ap.add_argument('-f','--filter',metavar='filter', type=str, nargs='?', default='standard', help='Input filter to use, see manual.')
	if len(sys.argv) > 1:
		args = ap.parse_args()
		t = MarkovishTranslator(args.model[0], args.input_file, args.input_column[0], args.output_file)
		t.setInputFilter(args.filter)
		#t.runModels()
		t.transcribe()
	else:
		t = MarkovishTranslator(open('myModel2.model'),open('../csv/dispensations_ocd.csv'),'doser',open('testparsed_ocd.csv','w'))
		op = t.transcribe()
		#t.writeOutput()
		#open('testparsed_ocd.csv','w').write(op)
		quit()	
		b = OperatorMarkovishTranslator(open('myOperatorModel2.model'),open('../csv/dispensations_training_sample.csv'),'doser',False)
		next(b)
		nOP = 0
		nOPcorrect = 0
		for x in op:
			if len(x) > 1:
				transit = ' '.join(b.appendOperators(b.get('doser'),x))
				print(b.get('doser')+' | '+' '.join(x) + '|'+transit)
				trans = ''
			if b.get('pattern') != '':
				trans = rmExtraSpace(stripArguments(b.get('pattern').lower()).replace('/','').replace('s', '')).strip()
				transit = rmExtraSpace(transit.replace('s','').replace('  ',' ')).strip().rstrip(';')
				nOP += 1
				if trans == transit: nOPcorrect += 1
				print('With op: '+trans+' predicted: ' +transit)
			print(trans)
			next(b)
		print(str(round(100*nOPcorrect/nOP,2))+'% of dispensations yielded a dosage value.')
		#t.writeOutput()
		#open('testparsed.csv','w').write(op)
		
