class Transcript:
    def __init__(self, wordList):
        """
        Instantiate by passing a list of tuples containing at least [0:('q',#), 1:('f',#)] but may contain f and o
        """
        self.wordList = wordList
        self.position = 0
        self.n = len(self.wordList)
    
    def __iter__(self):
        self.position = 0
        return self
     
    def __next__(self):
        self.position += 1
        if self.position > self.n: raise StopIteration
        return self.wordList[self.position-1]
    
    def rewind(self):
        self.position = 0
        return self
    
    def totals(self):
        """
        Calculate total weight at each position (q or q*f)
        """
        s = {}
        valPos = -1
        for p,x in enumerate(self):
            key, val = x
            if key == 'q':
                s[p] = val
                valPos = p
            elif key == 'f' and valPos > -1:
                s[valPos] = s[valPos] * val
        return s.values()
    
    def objects(self):
        s = []
        for p,x in enumerate(self):
            key, val = x
            if key == 'o':
                s += [val]
        return ','.join(s)
    
    def calculation(self):
        """
        Return a string representation of the calculation performed
        """
        s = ''
        for p,x in enumerate(self):
            key, val = x
