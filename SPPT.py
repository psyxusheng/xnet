# -*- coding:utf8  -*-
from functools import reduce
from numpy import ndarray,sqrt
import re
import tensorflow as tf

def build_embTable(vocab):
    with tf.variable_scope('embedding',reuse= tf.AUTO_REUSE):
        if isinstance(vocab , ndarray):
            num_tok,emb_dim = vocab.shape
            table = tf.get_variable(name = 'embedding_table',
                                    shape = [num_tok,emb_dim],
                                    initializer=tf.constant_initializer(value = vocab,dtype=tf.float32,
                                                                        verify_shape=True),
                                    trainable = True)
        else:
            num_tok,emb_dim = vocab
            initializer = tf.truncated_normal_initializer(stddev =  sqrt(1/emb_dim))
            table = tf.get_variable(name = 'embedding_table',
                                    shape = [num_tok,emb_dim],
                                    initializer=initializer,
                                    dtype = tf.float32,
                                    trainable= True)
    return table

def lookup(table,value):
    table_ = tf.concat([tf.zeros_like(table[0:1,:]),table[1:,:]],axis=0,name='skipZero')
    return tf.nn.embedding_lookup(table_,value)
    

def bilstm(net,lengths,lstm_units,training,keep_prob):
    
    Fcells,Bcells = [],[]
    for i,u in enumerate(lstm_units):
        with tf.variable_scope('biLSTMLayer%d'%(i+1)):
            fcell = tf.nn.rnn_cell.BasicLSTMCell(num_units= u//2,activation=tf.nn.tanh,
                                                 name = 'FCLayer%d'%(i+1))
            bcell = tf.nn.rnn_cell.BasicLSTMCell(num_units= u//2,activation=tf.nn.tanh,
                                                 name = 'BCLayer%d'%(i+1))
            if training:
                fcell = tf.nn.rnn_cell.DropoutWrapper(fcell,output_keep_prob=keep_prob)
                bcell = tf.nn.rnn_cell.DropoutWrapper(bcell,output_keep_prob=keep_prob)
            Fcells.append(fcell)
            Bcells.append(bcell)
    
    MultiFcell = tf.nn.rnn_cell.MultiRNNCell(Fcells)
    MultiBcell = tf.nn.rnn_cell.MultiRNNCell(Bcells)

    states,final = tf.nn.bidirectional_dynamic_rnn(cell_fw         = MultiFcell,
                                                   cell_bw         = MultiBcell,
                                                   inputs          = net,
                                                   sequence_length =lengths,
                                                   dtype           = tf.float32,
                                                   scope           = 'bilstmEnc')
    
    states_concated = tf.concat(states , axis=2, name = 'concat4bidir')
    return states_concated

def build_placeholders(batch_size = None,time_step =None):
    sentence = tf.placeholder(shape = [batch_size,time_step],
                              name = 'sentence',dtype=tf.int32)
    sentence_length = tf.placeholder(shape = [batch_size],
                                     name='sentence_length',
                                     dtype = tf.int32)
    sentence_dash = tf.placeholder(shape = [batch_size,time_step],
                                   name = 'sentence_dash',dtype=tf.int32)
    sentence_dash_length = tf.placeholder(shape = [batch_size],
                                          name='sentence_dash_length',
                                          dtype = tf.int32)
    pair_target          = tf.placeholder(shape=[batch_size],
                                          dtype=tf.float32,
                                          name = 'pair_target')
    target               = tf.placeholder(shape = [batch_size],
                                          dtype=tf.int32,
                                          name = 'target')
    return sentence,sentence_length,sentence_dash,sentence_dash_length,pair_target,target

def max_pooling(net):
    return tf.reduce_max(net,axis=1,name='max_pooling')

def modelSize():
    ret = 0
    for v in tf.trainable_variables():
        ret+= reduce(lambda x,y:x*y, v.shape)
    return int(ret)

def build_model(vocab,lstm_units=[128,256,128],
                target_cls = 1400,
                training=True,keep_prob=.5,
                lr_schedule = [1e-2,0.9,2000],
                batch_size=72,time_step=100,
                pt_layers  = ['embedding','FCLayer1','FCLayer2',
                              'BCLayer1','BCLayer2'],
                ft_rate    = 0.01,):
    print('building model ...')
    reg = re.compile("|".join(pt_layers))
    init_lr,decay_rate,decay_step=lr_schedule
    sent,sent_len,sentdash,sentdash_len,ptarget,target = build_placeholders(batch_size,time_step)
    table = build_embTable(vocab)
    with tf.variable_scope('sent',reuse = tf.AUTO_REUSE):
        sent_emb    = lookup(table,sent)
        sent_bilstm = bilstm(sent_emb,sent_len,lstm_units,training,keep_prob)
        sent_enc    = max_pooling(sent_bilstm)
    with tf.variable_scope('sent_dash',reuse=tf.AUTO_REUSE):
        sent_d_emb    = lookup(table,sentdash)
        sent_d_bilstm = bilstm(sent_d_emb,sentdash_len,lstm_units,training,
                               keep_prob)
        sent_d_enc    = max_pooling(sent_d_bilstm)
    with tf.variable_scope('pretrain',reuse=tf.AUTO_REUSE):
        cross_feat = tf.concat([sent_enc,sent_d_enc,
                                tf.abs(sent_enc-sent_d_enc),
                                tf.multiply(sent_enc,sent_d_enc)],axis=-1,
                                name = 'cross_feat')
        plogits = tf.squeeze(tf.layers.dense(cross_feat,units=1,reuse=tf.AUTO_REUSE,
                                             activation = None,name='plogits'),
                             axis = 1,name='psqueeze')
        ppred   = tf.nn.sigmoid(plogits,name='pprediction')
        
        pcost   = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = ptarget,
                                                                     logits = plogits),
                                 name = 'pcost')
        pgs = tf.Variable(initial_value=0,trainable=False,
                          name='pgs',dtype=tf.int32)
        plr = tf.train.exponential_decay(learning_rate = init_lr,
                                          global_step  = pgs,
                                          decay_steps  = decay_step,
                                          decay_rate   = decay_rate,
                                          staircase=True,
                                          name='plr')
        
        popt = tf.train.AdamOptimizer(learning_rate=plr,
                                      name = 'poptimizer')
        grads, variables = zip(*popt.compute_gradients(pcost))
        grads, _   = tf.clip_by_global_norm(grads,10)
        ptrainOp = popt.apply_gradients(zip(grads, variables),global_step=pgs,
                                            name = 'ptrainOp')
    with tf.variable_scope('finetune',reuse=tf.AUTO_REUSE):
        flogits = tf.layers.dense(sent_enc,units=target_cls,activation=None,
                                  name = 'floits')
        fpred  = tf.nn.softmax(flogits,axis=-1)
        fcost   = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = flogits,
                                                                                labels = target))
        fgs = tf.Variable(initial_value=0,trainable=False,
                          name='fgs',dtype=tf.int32)
        flr = tf.train.exponential_decay(learning_rate = init_lr,
                                          global_step  = fgs,
                                          decay_steps  = decay_step,
                                          decay_rate   = decay_rate,
                                          staircase=True,
                                          name='flr')
        fopt = tf.train.AdamOptimizer(learning_rate=flr,
                                      name = 'foptimizer')
        rescaled_grads=[]
        rescaled_vars =[]
        grads_and_vars = fopt.compute_gradients(fcost)
        for i,(grad,var) in enumerate(grads_and_vars):
            if isinstance(grad,type(None)):
                continue
            rescaled_vars.append(var)
            if reg.search(var.name):
                rescaled_grads.append(grad * ft_rate)
            else:
                rescaled_grads.append(grad)
        ftrainop = fopt.apply_gradients(zip(rescaled_grads,rescaled_vars), 
                                        global_step = fgs,name = 'ftrainop')
        
        
    pretrain_objs = [[sent,sent_len,sentdash,sentdash_len,ptarget],[plogits,ppred,pcost,ptrainOp]]
    finetune_objs = [[sent,sent_len,target],[flogits,fpred,fcost,ftrainop]]
    print('finish building model with parameter size %.3e'%modelSize())
    return pretrain_objs,finetune_objs


class Builder():
    def __init__(self,**kwargs):
        pretrain_objs,finetune_objs = build_model(**kwargs)
        self.pt_placeholders,[self.plogit,self.ppred,self.pcost,self.ptrainop] = pretrain_objs
        self.ft_placeholders,[self.flogit,self.fpred,self.fcost,self.ftraonop] = finetune_objs
    def zip_data(self,data,phase):
        if phase == 'pretrain':
            feed_in = dict(zip(self.pt_placeholders,data))
        elif phase == 'finetune':
            feed_in = dict(zip(self.ft_placeholders,data))
        return feed_in
    @property
    def pt(self,):
        return self.plogit,self.ppred,self.pcost,self.ptrainop
    @property
    def ft(self,):
        return self.flogit,self.fpred,self.fcost,self.ftraonop
    



if __name__=='__main__':
    tf.reset_default_graph()
    model= Builder(vocab=[2200,32])