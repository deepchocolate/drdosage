import re
from num import isNumeric;
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

def simplify(txt, stripChars='.?*=-+"!,\n'):
	"""
	Lowercase, remove extra whitespace and "unnecessary" characters at
	tails of words in string txt.
	"""
	return polish(rmExtraSpace(txt.lower()), stripChars)

REPLACOR = re.compile('[0-9]')
# Fix this and make a test of each word whether it's numeric or not
def replaceNumbers(txt, replacement='#'):
	"""
	Replace numbers in string txt with replacement.
	"""
	return ' '.join(['#' if isNumeric(x) else x for x in txt.split(' ')])
