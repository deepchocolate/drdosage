# Remove repeated spaces
import re
def rmExtraSpace(txt):
	while txt.find('  ') != -1: txt = txt.replace('  ', ' ')
	return txt

def simplify(txt, stripChars='.?*=-+"!,\n'):
	"""
	Remove "unnecessary" characters at tails of words and extra whitespace.
	"""
	words = rmExtraSpace(txt).lower().split(' ')
	o = ''
	for w in words: o += w.strip(stripChars) + ' ' 
	return rmExtraSpace(o.strip())

REPLACOR = re.compile('[0-9]')
def replaceNumbers(txt, replacement='#'):
	return REPLACOR.sub(replacement, txt)
