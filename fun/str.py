# Remove repeated spaces
def rmExtraSpace(txt):
    while txt.find('  ') != -1: txt = txt.replace('  ', ' ')
    return txt

def simplify(txt, stripChars='.?*=-+"!,'):
    words = rmExtraSpace(txt).lower().split(' ')
    o = ''
    for w in words: o += w.strip(stripChars) + ' ' 
    return rmExtraSpace(o.strip())
