import pickle
import numpy as np

def SaveObj(obj, path):
    if '.pkl' not in path: raise ValueError('file is not .pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def LoadObj(path):
    if '.pkl' not in path: raise ValueError('file is not .pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)

def recur_items(dictionary,nest):
    for key, value in dictionary.items():
        if type(value) is dict:
            yield (key,nest)
            yield from recur_items(value,nest+1)
        else:
            yield (key,nest)

def print_dic(dic,nest=0):
    for key,nest in recur_items(dic,0):
        print("\t"*nest,key)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def movingaverage(interval, window_size, mode):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, mode)

def printTitle(title):
    print("="*np.max([0,np.int((60-len(title))/2)+(60-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((60-len(title))/2)]))
    print("\n")

def printAttr(attr, skip=[]):
    for k in attr:
        if k in skip : continue
        print("\t",k,"-"*(1+len(max(attr,key=len))-len(k)),">",attr[k])
    print("\n")
