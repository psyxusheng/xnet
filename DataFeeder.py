# -*- coding: utf-8 -*-
import os
from tqdm import trange
import numpy as np
from Vocab import Vocab

vocab = Vocab('vocab.whole.char.csv')
sep = vocab.vocab['[sep]']


def split_kc(kcs_str):
    kcs_tuple = [kc.split(';') for kc in kcs_str.split(',')]
    kcs,is_mains = list(zip(*kcs_tuple))
    return [int(k) for k in kcs]

def loadCorpus(fn):
    """
    corpus is in format item \t anwser \t diffcult level \t kc;is_main,
    """
    data = []
    for line in open(fn,'r',encoding='utf8'):
        try:
            item,dscp,dl,kcs_str = line.strip().split('\t')
            if len(dscp.strip())<10:
                continue
            kcs = split_kc(kcs_str)
            data.append([item,dscp,int(dl),kcs])
        except:
            continue
    return data

def load_and_convert2id(fn):
    data = loadCorpus(fn)
    for i,(item,dscp,dl,kcs) in enumerate(data):
        itemIds = vocab.sentence2id(item)
        dscpIds = vocab.sentence2id(dscp)
        data[i][0] = itemIds
        data[i][1] = dscpIds
    return data

def random_sample(a,b):
    while True:
        sampled = np.random.choice(a)
        if sampled != b:
            break
    return sampled
        

class DataFeeder():
    def __init__(self,fn):
        self.data = load_and_convert2id(fn)
        self.dataSize = self.data.__len__()
        self.shuffle()
    def shuffle(self,):
        np.random.shuffle(self.data)
        self.cursor = 0
    def next_batch_pair_task(self,batch_size):
        items,item_lengths = [],[]
        dscps,dscp_lengths = [],[]
        pairs              = []
        for rid in np.random.choice(self.dataSize,replace=False,size = batch_size):
            itm,dsp,dl,kc = self.data[rid]
            pair = 1
            if np.random.random()<=0.5:
                dsp = self.data[random_sample(self.dataSize,rid)][1]
                pair = 0
            items.append(itm)
            item_lengths.append(len(itm))
            dscps.append(dsp)
            dscp_lengths.append(len(dsp))
            pairs.append(pair)
            
        max_item_seqlen = max(item_lengths)
        max_dscp_seqlen = max(dscp_lengths)
        
        item_array = np.zeros([batch_size,max_item_seqlen],dtype=np.int32)
        dscp_array = np.zeros([batch_size,max_dscp_seqlen],dtype=np.int32)
        item_lengths = np.array(item_lengths)
        dscp_lengths = np.array(dscp_lengths)
        pair         = np.array(pairs,dtype=np.float32)
        for i in range(batch_size):
            item_array[i,:item_lengths[i]] = items[i]
            dscp_array[i,:dscp_lengths[i]] = dscps[i]
        return item_array,item_lengths,dscp_array,dscp_lengths,pair
    def next_batch_predict_task(self,batch_size):
        items,lengths,kcs,kc_num=[],[],[],[]
        for rid in np.random.choice(self.dataSize,replace=False,size = batch_size):
            itm,dsp,dl,kc = self.data[rid]
            items.append(itm)
            lengths.append(len(itm))
            kcs.append(kc)
            kc_num.append(len(kc))
        max_kc_num = max(kc_num)
        max_seqlen = max(lengths)
        
        kc_array=np.ones([batch_size,max_kc_num],dtype=np.int32)*-1
        item_array = np.zeros([batch_size,max_seqlen],dtype=np.int32)
        lengths = np.array(lengths,dtype=np.int32)
        for i in range(batch_size):
            item_array[i,:lengths[i]] = items[i]
            kc_array[i,:kc_num[i]] = kcs[i]
        return item_array,lengths,kc_array
        
        
        
        
if __name__ == '__main__':
    data = DataFeeder('PairedCorpus.test.txt')
    for i in trange(1000):
        x = data.next_batch_predict_task(100)