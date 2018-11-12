#import num,str
from .num import *
from .transcript import *
class Transcripts:
    def __init__(self,outputFormat = 'q,f,o'):
        self.ignore = {}
        #self.outputFormat = {x:0 for x in outputFormat.split(',')}
        self.outputFormat = {'q':0, 'f':0, 'o':''}
        # Container for manually classified strings
        #self.classified = {}
        self.ignoreChar = '/'
        # Number of transcripts
        self.n = 0
        # Patterns
        #self.patterns = {}
        # Map from word to key, eg "tablett" -> O by its position in the sentence
        # {1: { 'doser':'O', 'ett':'Q' } }
        self.wordToKey = {}
        # Translating pattern to values
        # Ex: {'QO//': {'q':{ 'ett' : 1, 'två' : 2 }}}
        # Ex: {1:{'q':{'en':1,'två':2},'o':{'tablett','t'}}}
        self.keyToVal = {}
        # Parsing functions
        #self.parsers = {'q':self.parseQ,'o':self.parseO,'q-f':self.parseQF,'/':self.addIgnore}
    
    def getTranscript(self,txt,addVal=True,sep=' '):
        s = ''
        i = 0
        #print(txt,self.wordToKey,self.keyToVal)
        for word in txt.split(' '):
            w = word.rstrip('.')
            sym = self.getKey(w,i)
            #print(sym)
            s += sym
            if addVal and sym != '/' and i in self.wordToKey and w in self.wordToKey[i] and self.wordToKey[i][w] in self.keyToVal[i] and w in self.keyToVal[i][self.wordToKey[i][w]]:
                s += str(self.keyToVal[i][self.wordToKey[i][w]][w])
            s += sep
            i += 1
        if s.strip('/ ') == '': return False
        else: return s.strip(' ')
    
    def getTranslation(self, txt, instruction=False):
        """
        Get translation according to desired output format
        """
        if instruction == False: instruction = self.getTranscript(txt,False, '')
        o = []
        word = txt.split(' ')
        for i in range(0,len(instruction)):
        #for i,word in enumerate(txt.split(' ')):
            w = word[i].rstrip('.')
            #k = self.getKey(w, i)
            k = instruction[i]
            if k != '/':
                v = self.getVal(w, i, k)
                if isinstance(v, bool): raise Exception('Not able to translate!')
                o += [(k, v)]
        if len(o) > 0: return Transcript(o)
        else: return False
            
    
    def getInstruction(self,txt,pattern=False,error=False):
        if pattern == False:
            pattern = self.getTranscript(txt)
            if pattern == False: pattern = '-'
        if error: print('The instruction could not be parsed! Please try again!')
        print('I could not understand some of this:')
        print(txt)
        print("Q [#] - Quantity\nO [name] - Object\n- to skip")
        print('Suggested: '+pattern.upper())
        inp = input("Enter details (leave blank to use suggestion): ")
        inp = inp.lower().strip()
        # Skip this text
        if inp == '-': return False
        elif len(inp) > 0:
            try:
                return self.parseInstruction(txt, inp)
            except BaseException as e:
                return self.getInstruction(txt, pattern, True)
        # If input empty, take the suggestion
        else: return self.getTranscript(txt, False, '')
    
    # Get key of word
    def getKey(self,word, pos):
        #print(word,pos)
        if isNumeric(word): word = '#'
        if pos in self.wordToKey and word in self.wordToKey[pos]:
            return self.wordToKey[pos][word]
        else: return '/'
    
    def getVal(self, word, pos, key):
        if (key == 'q' or key == 'f') and isNumeric(word): return float(word)
        if word in self.keyToVal[pos][key]: return self.keyToVal[pos][key][word]
        else: return False
    
    def parseInstruction(self, text, instruction):
        """
        Parse instruction applied to text and return the raw instruction with values stripped
        """
        s = ''
        textParts = text.split(' ')
        i = 0
        for word in instruction.split(' '):
            if word != '/':
                tp = textParts[i].rstrip('.')
                # If word is not number, use instruction
                #print(tp, word)
                self.addWordToKey(i, tp, word[0])
                # Single q in instruction implies word is numeric 
                if len(word) > 1: self.addKeyToVal(i, tp, word[0], word[1:])
                # Test the validity of text as numeric
                elif word[0] != 'o' and word[0] != '/': float(tp)
                #i += 1
                s += word[0]
            else:
                s += '/'
            i += 1
        return s
        
    # Add a pattern txt to position pos (in sentence) of type obj
    def addWordToKey(self,pos,txt, obj):
        if isNumeric(txt): key = '#'
        else: key = txt
        if pos in self.wordToKey: self.wordToKey[pos][key] = obj
        else: self.wordToKey[pos] = {key:obj}

    # key: key of pattern. txt: word to translate. obj: object type. val: what object
    def addKeyToVal(self,pos,txt,obj,val):
        if obj == 'f' or obj == 'q': val = float(val)
        if not pos in self.keyToVal: self.keyToVal[pos] = {obj : {txt:val}}
        elif not obj in self.keyToVal[pos]: self.keyToVal[pos][obj] = {txt:val}
        else: self.keyToVal[pos][obj][txt] = val
    
    def addIgnore(self,pos,word):
        if not pos in self.ignore: self.ignore[pos] = [word]
        elif not word in self.ignore[pos]: self.ignore[pos] += [word]
    
    def getObjectData(self):
        return {'wordToKey':self.wordToKey, 'keyToVal':self.keyToVal}
    
    def setObjectData(self,data):
        """
        Data should be dict as returned by getObjectData
        """
        #print(data['wordToKey'])
        #data['wordToKey'] = {int(k):v for k,v in data['wordToKey'].items()}
        #print(data['wordToKey'])
        #quit()
        # Annoyingly, using JSON, keys have to be converted back to integer in map
        self.wordToKey = {int(k):v for k,v in data['wordToKey'].items()}
        self.keyToVal = {int(k):v for k,v in data['keyToVal'].items()}
        #print(self.wordToKey)
        #print(self.keyToVal)
        #quit()
        #self.wordToKey = data['wordToKey']
        #self.keyToVal = data['keyToVal']
        return self
    
    def toJSON(self):
        return json.dumps(self.getObjectData())
