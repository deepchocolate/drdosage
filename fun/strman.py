import re
from .num import isNumeric
# Remove repeated spaces
def rmExtraSpace(txt):
	while txt.find('  ') != -1: txt = txt.replace('  ', ' ')
	return txt

def polish(txt, chars='.?*=-+"!,\n'):
	"""
	Remove "unnecessary" characters at tails of words and extra whitespace.
	"""
	words = [x.strip(chars) for x in txt.split(' ')]	
	return ' '.join(words)

# Regular expressions used by separatechars()
RE_SEPLEADINTFROMSTR = re.compile('[0-9]+[A-Za-z]')
RE_SEPWORDS = re.compile('[a-z][\.,][a-z]')

def separateChars(txt, regExpr):
	"""
	Separate characters in string txt depending on the regular expression regExpr. 
	"""
	found = regExpr.search(txt)
	if found != None:
		st = found.start()
		ed = found.end() - 1
		return separateChars(txt[0:ed] + ' ' + txt[ed:], regExpr)
	else: return txt

def separateLeadIntFromStr(txt):
	return separateChars(txt, RE_SEPLEADINTFROMSTR)

def separateWords(txt):
	return separateChars(txt, RE_SEPWORDS)

def replaceSubStr(txt, target, replacement, regExpr):
	"""
	Replace replacement with target in txt where regular expression regExpr matches.
	"""
	found = regExpr.search(txt)
	if found != None:
		st = found.start()
		ed = found.end()
		return replaceSubStr(txt[:st]+txt[st:ed].replace(target, replacement)+txt[ed:], target, replacement, regExpr)
	else: return txt

RE_COMMASINNUM = re.compile('[0-9],[0-9]')
RE_ADDSTR = re.compile(' ?[0-9]+(\+[0-9])+ ')
def addUpNumbers(txt):
	found = RE_ADDSTR.search(txt)
	if found != None:
		st = found.start()
		ed = found.end()
		return addUpNumbers(txt[:st]+str(eval(txt[st:ed]))+' '+txt[ed:])
	else: return txt

def replaceCommasInNumbers(txt, target=',', replacement='.') -> str:
	return replaceSubStr(txt, target, replacement, RE_COMMASINNUM)

def replaceHalves(txt):
	return txt.replace('en halv ', '0.5 ')
	
def simplify(txt, stripChars=')(.?*=-+"!,\n'):
	"""
	Lowercase, remove extra whitespace and "unnecessary" characters at
	tails of words in string txt.
	"""
	return polish(rmExtraSpace(txt.lower()), stripChars)

def lowercaseText(txt):
	return txt.lower()

def replaceNumbers(txt, replacement='#'):
	"""
	Replace numbers in string txt with replacement.
	"""
	return ' '.join(['#' if isNumeric(x) else x for x in txt.split(' ')])

INPUT_FILTERS = {
	'standard':[lowercaseText, rmExtraSpace, separateWords, separateLeadIntFromStr, polish],
	'extended':[lowercaseText, addUpNumbers,rmExtraSpace,replaceHalves, separateWords, separateLeadIntFromStr, polish, replaceCommasInNumbers]
}
