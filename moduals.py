# -*- coding: utf-8 -*-
from numpy import ndarray
from math import sqrt
import tensorflow as tf


def self_attention_pooling(net,scope=''):
    """
        net is in shape [batch_size, time_steps , dim]
        
    """
    dim = net.get_shape().as_list()[-1]
    initializer = tf.truncated_normal_initializer(stddev = 1/sqrt(dim))
    with tf.variable_scope(scope+'/SeflAttentionWeight',reuse = tf.AUTO_REUSE):
        """calc weight"""
        saw = tf.get_variable(name = 'self_attention_weight',
                              shape = [dim,dim],
                              dtype = tf.float32,
                              initializer = initializer)
        
        weight = tf.tensordot(net,saw,axes = [2,0])
        weight = tf.matmul(weight,tf.transpose(net,[0,2,1]))
        weight = tf.reduce_sum(weight,axis=-1,keepdims=True)
        """masking"""
        sums = tf.reduce_sum(tf.abs(net) , axis = -1 ,keepdims=True)
        mask_values = tf.ones_like(sums) * (-2**32+1)
        weight = tf.where(tf.equal(sums,0.),mask_values,weight)
        weight = tf.nn.softmax(weight,axis=1)
        values = tf.matmul(tf.transpose(weight,[0,2,1]),net)
        
        return tf.squeeze(values,axis=1),tf.squeeze(weight,axis=2)

def max_pooling(net,scope=''):
    with tf.variable_scope(scope+'/max_pooling'):
        return tf.reduce_max(net,axis=1),tf.constant(0.0)
    
def mean_pooling(net,scope=''):
    with tf.variable_scope(scope+'/mean_pooling'):
        return tf.reduce_mean(net,axis=1),tf.constant(0.0)
    
def bilstm(net,lengths,lstm_units,training,keep_prob,scope=''):
    with tf.variable_scope(scope+'/bilstmEncoder',reuse = tf.AUTO_REUSE):
        Fcells,Bcells = [],[]
        for i,u in enumerate(lstm_units):
            fcell = tf.nn.rnn_cell.BasicLSTMCell(num_units= u//2,activation=tf.nn.tanh,
                                                 name = 'lstmForewardCell%d'%(i+1))
            bcell = tf.nn.rnn_cell.BasicLSTMCell(num_units= u//2,activation=tf.nn.tanh,
                                                 name = 'lstmBackwardCell%d'%(i+1))
            if training:
                fcell = tf.nn.rnn_cell.DropoutWrapper(fcell,output_keep_prob=keep_prob)
                bcell = tf.nn.rnn_cell.DropoutWrapper(bcell,output_keep_prob=keep_prob)
            Fcells.append(fcell)
            Bcells.append(bcell)
        MultiFcell = tf.nn.rnn_cell.MultiRNNCell(Fcells)
        MultiBcell = tf.nn.rnn_cell.MultiRNNCell(Bcells)
        states,final = tf.nn.bidirectional_dynamic_rnn(cell_fw = MultiFcell,
                                                       cell_bw = MultiBcell,
                                                       inputs  = net,
                                                       sequence_length=lengths,
                                                       dtype=  tf.float32,
                                                       scope = 'bilstmEncoding')
        states_concated = tf.concat(states , axis=2, name = 'concat4bidir')
    return states_concated

        
def multi_layer_perception(net,num_units,training,keep_prob,scope=''):
    with tf.variable_scope(scope+'/multiLayerPerception',reuse = tf.AUTO_REUSE):
        net = tf.layers.dense(net,num_units[0],activation=tf.nn.relu,name = 'hiddenLayer')
        net = tf.layers.dropout(net,rate=keep_prob,training = training)
        net = tf.layers.dense(net,num_units[1],activation=tf.nn.relu,name = 'outputLayer')
    return net

def build_embeddingTable(vocab,scope=''):
    with tf.variable_scope(scope+'/embeddingTable',reuse = tf.AUTO_REUSE):
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

def lookup(table,value,scope=''):
    with tf.variable_scope(scope+'/embedding_lookup'):
        table_ = tf.concat([tf.zeros_like(table[0:1,:]),table[1:,:]],axis=0,name='skipZero')
        return tf.nn.embedding_lookup(table_,value)



if __name__ == '__main__':
    tf.reset_default_graph()
    vocab = [2200,200]
    table = build_embeddingTable(vocab)
    x = tf.placeholder(name = 'tester',shape=[None,None],dtype=tf.int32)
    l = tf.placeholder(name = 'lengths',shape=[None],dtype=tf.int32)
    y = lookup(table,x)
    z = bilstm(y,l,[256,256],True,0.5)
    w = self_attention(z)
    u = max_pooling(z)



