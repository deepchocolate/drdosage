# Remove repeated spaces
def rmExtraSpace(txt):
    while txt.find('  ') != -1: txt = txt.replace('  ', ' ')
    return txt
