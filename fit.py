from cla import *
import numpy as np
from scipy.optimize import minimize
f = FileIO(open('test.csv'))
c = Classifier()
c.includedPatterns = [1,2,3]
c.y = []
c.x = []
def logit(x):
    return 1/(1 + np.exp(-x))

def dLogitOutput(x):
    return x*(1 - x)
    
def softmax (x):
    return np.exp(x)/(np.sum(np.exp(x),1))
"""
def calcError (x, out,inp, neurons):
	xlen = len(inp[0])
	#print(len(x))
	#print(x[:xlen*neurons])
	#print(x[xlen*neurons:])
	wx = np.array(x[:xlen*neurons]).reshape(xlen,neurons)
	wy = np.array(x[xlen*neurons:]).reshape(neurons,len(out[0]))
	#print(wx)
	#print(wy)
	ypred = logit(np.dot(logit(np.dot(inp,wx)),wy))
	error = out-ypred
	return sum(sum(error*error))
"""
#def predict(x,wx,wy):
#	return logit(np.dot(logit(np.dot(x,wx)),wy))

while next(f) != False:
        c.y += [[int(f.get('V1')),int(f.get('V2')),int(f.get('V3'))]]
        c.x += [[1,int(f.get('x'))]]
        #c.x += [[1]]

c.hiddenNeurons = 3
"""
wx, wy = c.createLayers()
a = wx.flatten().tolist()
b = wy.flatten().tolist()
x = a + b
#print(c.hiddenNeurons)
#print(wx)
#print(a)
#print(a.reshape(len(c.x[0]),c.hiddenNeurons))

m = minimize(fun=calcError, x0=x,args=(c.y,c.x,3),method='L-BFGS-B')
wx = np.array(m.x[:2*3]).reshape(2,3)
wy = np.array(m.x[2*3:]).reshape(3,3)
print(np.mean(predict(c.x,wx,wy),0))
"""
c.updateNetwork3()
print(np.mean(c.predict(c.x,c.synapse[0],c.synapse[1]),0))
quit()
for i in iter(range(3,4)):
    print('Neurons',i)
    c.hiddenNeurons = i
    #c.updateNetwork()
    #print(c.logit(np.dot(c.logit(c.synapse[0]),c.synapse[1])))
    #ms = np.dot(logit(np.dot(c.x,c.synapse[0])),c.synapse[1])
    #print('Last mean 1', logit(np.mean(ms,0)))
    #print('Last softmax 1',(np.mean((np.exp(ms).T/(np.sum(np.exp(ms),1))).T,0)))
    s1,s2 = c.updateNetwork2()
    ms = np.dot(logit(np.dot(c.x,s1)),s2)
    print('Last mean 2', logit(np.mean(ms,0)))
    print('Last softmax 2',(np.mean((np.exp(ms).T/(np.sum(np.exp(ms),1))).T,0)))
    #ms = np.dot(logit(np.dot(c.x,c.minimum[0])),c.minimum[1])
    #print('Minimum mean', logit(np.mean(ms,0)))
    #print('Minimum softmax',(np.mean((np.exp(ms).T/(np.sum(np.exp(ms),1))).T,0)))
    #print(np.exp(ms)/np.sum(np.exp(ms)))
    #print(np.dot(c.logit(np.dot(c.x,c.synapse[0])),c.synapse[1]))
    #print(c.logit(np.dot(c.logit(c.synapse[0]),c.synapse[1])))
    #print('Minmat',c.minimum,'othermat',c.synapse)
c.writeErrors()
#print(c.logit(np.dot(c.logit(np.dot(c.x,c.synapse[0])),c.synapse[1])))
#print(c.logit(np.dot(c.logit(c.synapse[0]),c.synapse[1])))

