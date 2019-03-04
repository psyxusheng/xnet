# -*- coding: utf-8 -*-
import numpy as np
import csv

def loadVocab(fn):
    fo = open(fn,'r',encoding='utf8')
    reader = csv.reader(fo)
    vectors=[]
    vocab = {}
    for log in reader:
        tok,ind,*vec = log
        vocab[tok] = int(ind)
        vectors.append(vec)
    vectors = np.array(vectors,dtype=np.float32)
    return vocab,vectors

class Vocab():
    def __init__(self,fn):
        self.vocab,self.vectors = loadVocab(fn)
        self.unk = self.vocab['[unk]']
        self.seq = self.vocab['[sep]']
    def sentence2id(self,sentence,max_length=200):
        chars = list(sentence)
        idxs  = [self.vocab.get(c,self.unk) for c in chars if c.strip()!='']
        return idxs[-max_length:]

if __name__ == '__main__':
    vocab = Vocab('vocab.whole.char.csv')
    y = vocab.sentence2id('我很开心的')