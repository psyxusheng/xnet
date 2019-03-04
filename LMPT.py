# -*- codingï¼š utf8  -*-
from functools import reduce
from numpy import ndarray,sqrt

import tensorflow as tf
import moduals

class PreTrainModel():
    def __init__(self,vocab,lstm_units,target_cls=1400,poome='max',training=True,keep_prob=.5,
                 lr_schedule = [1e-4,0.9,2000],fine_tune_rate = 1e-3, time_step = None , batch_size = None):
        """
            have two phase : representation learning phrase and task-specific fine-tune
                implemented by using different learning rates
            args:
                pm: pooling method, only support 'max pooling' and 'self attention pooling'
        """
        with tf.variable_scope('placeholders'):
            
            with tf.variable_scope('RL'):
                
                sentence = tf.placeholder(shape = [batch_size,time_step],
                                           name = 'sentence',dtype=tf.int32)
                sentence_length = tf.placeholder(shape = [batch_size],name='sentence_length',
                                                 dtype = tf.int32)
            with tf.variable_scope('FT'):
                
                ft_target = tf.placeholder(shape=[batch_size,None],
                                           name = 'targets',dtype=tf.float32)
            
        LM_inputs  = sentence[:,:-1]
        LM_targets = sentence[:,1:]
        lengths = sentence_length - 1
        
        with tf.variable_scope('RL'):
            embTable = moduals.build_embeddingTable(vocab,scope='sentence')
            num_tok,emb_dim = embTable.get_shape().as_list()
            
            """embedding layer """
            sent_embed = moduals.lookup(embTable,LM_inputs,scope='sentence')
            sent_embed = tf.layers.dropout(sent_embed,
                                           rate = keep_prob,training=training)            
            """ bi-lstm layer """
            sent_bilstm = moduals.bilstm(sent_embed,lengths,lstm_units,
                                         training,keep_prob,scope='sentence')
            sent_bilstm = tf.layers.dropout(sent_bilstm,rate = keep_prob , training = training)
            
        with tf.variable_scope('LM_task'):
            rl_logits = tf.layers.dense(sent_bilstm , units = num_tok,
                                        activation = tf.nn.relu,name='lm_logits')
        if poome == 'max':
            """max pooling """
            sent_encoded,sent_weight = moduals.max_pooling(sent_bilstm,scope='sentence')
        elif poome == 'mean':
            sent_encoded,sent_weight = moduals.mean_pooling(sent_bilstm,scope='sentence')
        elif poome == 'sap': 
            """self attention pooling"""
            sent_encoded,sent_weight = moduals.self_attention_pooling(sent_bilstm,scope='sentence')
        else:
            raise Exception('unsupport pooling method')
            
        with tf.variable_scope('FT'):
            sent_encoded = tf.layers.dropout(sent_encoded,rate=keep_prob,training=training)
            ft_logits = moduals.multi_layer_perception(sent_encoded,num_units = [target_cls // 2,target_cls],scope='ft_logits',
                                                       training=training,keep_prob=keep_prob)
        
        
        rl_prob = tf.nn.softmax(rl_logits)
        ft_prob = tf.nn.sigmoid(ft_logits)

        if training:
            
            rl_loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = rl_logits,
                                                                       labels = LM_targets)
            rl_cost   = tf.reduce_mean(rl_loss)
            
            ft_loss   = tf.nn.sigmoid_cross_entropy_with_logits(logits = ft_logits,
                                                                labels = ft_target)
            ft_cost   = tf.reduce_mean(ft_loss)
            
            with tf.variable_scope('training',reuse = False):
                init_lr,decay_rate,decay_steps = lr_schedule
                """for RL op RL layers using global lr"""
                with tf.variable_scope('RL'):
                    rl_gs = tf.get_variable(name = 'gs',shape=[],
                                            initializer=tf.constant_initializer(value=0),
                                            dtype=tf.int32,trainable=False)
                    rl_lr = tf.train.exponential_decay(learning_rate = init_lr,
                                                       global_step = rl_gs,
                                                       decay_steps = decay_steps,
                                                       decay_rate=decay_rate,
                                                       staircase=True,
                                                       name='lr')
                    rl_opt = tf.train.AdamOptimizer(learning_rate=rl_lr,
                                                    name = 'RL_optimizer')
                    grads, variables = zip(*rl_opt.compute_gradients(rl_cost))
                    grads, _   = tf.clip_by_global_norm(grads,10)
                    rl_trainOp = rl_opt.apply_gradients(zip(grads, variables),global_step=rl_gs,
                                                        name = 'rl_trainOp')

                with tf.variable_scope('FT'):
                    ft_gs = tf.get_variable(name = 'gs',shape=[],initializer=tf.constant_initializer(value=0),
                                         dtype=tf.int32,trainable=False)
                    ft_lr = tf.train.exponential_decay(learning_rate = init_lr,
                                                               global_step = ft_gs,
                                                               decay_steps = decay_steps,
                                                               decay_rate=decay_rate,
                                                               staircase=True,
                                                               name='lr')
                    ft_opt = tf.train.AdamOptimizer(learning_rate = ft_lr,name='FT_optimizer')
                    ft_grads,ft_vars = zip(*ft_opt.compute_gradients(ft_cost))
                    ft_grads,_ = tf.clip_by_global_norm(ft_grads,10)
                    rescaled_grads = []
                    rescaled_vars  = []
                    for i,grad in enumerate(ft_grads):
                        if isinstance(grad , type(None)):
                            continue
                        rescaled_vars.append(ft_vars[i])
                        if ft_vars[i].name.startswith('RL'):
                            rescaled_grads.append(grad * fine_tune_rate)
                        else:
                            rescaled_grads.append(grad)
                    ft_trainOp = ft_opt.apply_gradients(zip(rescaled_grads,rescaled_vars), 
                                                        global_step = ft_gs)
            self.rl_trainOp = rl_trainOp
            self.ft_trainOp = ft_trainOp
            self.rl_cost    = rl_cost
            self.ft_cost    = ft_cost
                    
        self.rl_placeholders = [sentence,sentence_length] 
        self.ft_placeholders = [sentence,sentence_length,ft_target]      
        self.rl_prob         = rl_prob
        self.ft_prob         = ft_prob
        self.rl_logits       = rl_logits
                
                
        '''mapping inputs placeholder into class's attributes'''

        #self.brief()
    def brief(self,):
        for v in tf.trainable_variables():
            print(v.name,v.shape)
    @property
    def size(self):
        ret = 0
        for v in tf.trainable_variables():
            ret += reduce(lambda x,y:x*y , v.shape)
        return int(ret) 
    
    
if __name__ == '__main__':
    tf.reset_default_graph()
    model = PreTrainModel(vocab = [2200,32],lstm_units=[128,256,128],
                          batch_size = 100 , time_step=72)