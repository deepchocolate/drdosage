#!/usr/bin/env python3
from cla.fileio import *
from fun import simplify
import argparse, sys
class Filter(FileIO):
    """
		It is recommended to put the filtered output in a new column, i.e. append it
		to a new column as the filtering may remove important semantics in the text.    
    """
    def __init__(self, inputStream, outputStream, inputColumn, outputColumn=False):
        # If not outputColumn, replace the data in this column.
        if outputColumn == False: outputColumn = inputColumn
        super(Filter,self).__init__(inputStream, outputStream)
        while next(self) != False:
            self.setCol(outputColumn, simplify(self.get(inputColumn)))
            self.setOutput()
        self.writeOutput()
   
	def removeExtraWhitespace(self):
		"""
		Collapse repeated whitespace into a single space ' '.
		"""
		pass

	def polishWords(self):
		"""
		Remove leading/trailing characters in words such as periods, question marks etc.
		"""
		pass
	
	def filter(self):
		"""
		Apply the selected filters to the input data.
		"""
		pass

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-ic', metavar='Column name',required=True, type=str, help='Column name to filter')
    ap.add_argument('-oc', metavar='Output column name', default=False, help='Name of column to place output. If omitted input column is replaced.')
    ap.add_argument('input', metavar='[input data]', type=argparse.FileType('r'), nargs='?', default=sys.stdin, help='File to read from, if omitted read from standard in.')
    ap.add_argument('output', metavar='[output data]',type=argparse.FileType('w'), nargs='?', default=sys.stdout, help='File to write to, if omitted output to standard out.')
    if len(sys.argv) > 1:
        args = ap.parse_args()
        filt = Filter(args.input, args.output, args.ic, args.oc)
        #print(args.input)
