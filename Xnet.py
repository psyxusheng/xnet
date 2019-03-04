# -*- coding： utf8  -*-
from functools import reduce
from numpy import ndarray,sqrt
import tensorflow as tf

class XNET():
    def __init__(self,vocab,tasks,lstm_units,pm='mp',training=True,keep_prob=.5,
                 lr_schedule = [1e-3,0.9,2000], time_step = None , batch_size = None):
        """
            args:
                pm: pooling method, only support 'max pooling' and 'self attention pooling'
        """
        with tf.variable_scope('embedTable',reuse = tf.AUTO_REUSE):
            if isinstance(vocab , ndarray):
                num_tok,emb_dim = vocab.shape
                embedTable = tf.get_variable(name = 'embedding',
                                         shape = [num_tok,emb_dim],
                                         initializer=tf.constant_initializer(value = vocab,dtype=tf.float32,
                                                                            verify_shape=True),
                                        trainable = True)
            else:
                num_tok,emb_dim = vocab
                initializer = tf.truncated_normal_initializer(stddev =  sqrt(1/emb_dim))
                embedTable = tf.get_variable(name = 'embedding',
                                            shape = [num_tok,emb_dim],
                                            initializer=initializer,
                                            dtype = tf.float32,
                                            trainable= True)
        def lookup(table,value,scope='embedding_lookup'):
            with tf.variable_scope(scope):
                table_ = tf.concat([tf.zeros_like(table[0:1,:]),table[1:,:]],axis=0,name='skipZero')
            return tf.nn.embedding_lookup(table_,value)
        
        with tf.variable_scope('input_placeholders'):
            sentences = tf.placeholder(shape = [batch_size,time_step],
                                       name = 'sentence_pair',
                                       dtype=tf.int32)
            targets   = []
            for i,(cls_num,type_,weight) in enumerate(tasks):
                if type_ == 'l':
                    #logistic result
                    targets.append(tf.placeholder(dtype=tf.int32,
                                                  shape = [batch_size,None],
                                                  name = 'task%d'%(i+1)))
                elif type_ == 's':
                    targets.append(tf.placeholder(dtype=tf.int32,
                                                  shape = [batch_size],
                                                  name = 'task%d'%(i+1)))
                else:
                    raise Exception('unsupported task type "%s"'%type_)
            
            lengths = tf.placeholder(name  = 'lengths',
                                     shape = [batch_size],
                                     dtype =tf.int32)
            
        if pm == 'sap':
            with tf.variable_scope('self_attention_weight',reuse = tf.AUTO_REUSE):
                dim = lstm_units[-1]
                initializer = tf.truncated_normal_initializer(stddev = sqrt(1/dim))
                saw = tf.get_variable(name = 'self_attention_weight',
                                      shape = [dim,dim],
                                      dtype = tf.float32,
                                      initializer=initializer)
        
        def self_attention(net,weight):
            """
                net is in shape [batch_size, time_steps , dim]
                
            """
            with tf.variable_scope('self_attentional_pooling'):
                """calc weight"""
                weight = tf.tensordot(net,weight,axes = [2,0])
                weight = tf.matmul(weight,tf.transpose(net,[0,2,1]))
                weight = tf.reduce_sum(weight,axis=-1,keepdims=True)
                """masking"""
                sums = tf.reduce_sum(tf.abs(net) , axis = -1 ,keepdims=True)
                mask_values = tf.ones_like(sums) * (-2**32+1)
                weight = tf.where(tf.equal(sums,0.),mask_values,weight)
                weight = tf.nn.softmax(weight,axis=1)
                values = tf.matmul(tf.transpose(weight,[0,2,1]),net)
                
                return tf.squeeze(values,axis=1),tf.squeeze(weight,axis=2)
                    
        def max_pooling(net,scope='max_pooling'):
            with tf.variable_scope(scope):
                return tf.reduce_max(net,axis=1),tf.constant(0.0)
            
        def bilstm(net,length,lstm_units,training,keep_prob,scope='bilstmEncoder'):
            with tf.variable_scope(scope,reuse = tf.AUTO_REUSE):
                Fcells,Bcells = [],[]
                for i,u in enumerate(lstm_units):
                    fcell = tf.nn.rnn_cell.BasicLSTMCell(num_units= u//2,activation=tf.nn.tanh,name = 'lstmForewardCell%d'%(i+1))
                    bcell = tf.nn.rnn_cell.BasicLSTMCell(num_units= u//2,activation=tf.nn.tanh,name = 'lstmBackwardCell%d'%(i+1))
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
        
        def mlp(net,num_unit,training,keep_prob,scope='mlp4logits'):
            dim = net.get_shape().as_list()[-1]
            with tf.variable_scope(scope,reuse = tf.AUTO_REUSE):
                net = tf.layers.dense(net,dim,activation=tf.nn.relu,name = 'hiddenLayer')
                net = tf.layers.dropout(net,rate=keep_prob,training = training)
                net = tf.layers.dense(net,num_unit,activation=tf.nn.relu,name = 'outputLayer')
            return net
                
        embed          = lookup(embedTable,sentences)
        lstm_encoded   = bilstm(embed,lengths,lstm_units,training,keep_prob)
        
        if pm == 'sap':
            """means self attention pooling"""
            representation,weights = self_attention(lstm_encoded,saw)
        elif pm == 'mp':
            """means max pooling"""
            representation,weights = max_pooling(lstm_encoded)
        else:
            raise Exception('unsupported pooling method "%s"'%pm)
        
        task_target_num,_,_ = list(zip(*tasks)) 
        
        logit_stacked = mlp(representation,
                            sum(task_target_num),
                            training,keep_prob)
        
        logits_splited        = tf.split(logit_stacked ,  
                                 num_or_size_splits= task_target_num,axis=-1)
        predictions  = []
        probablities = []
        for i,logits in enumerate(logits_splited):
            target_num,type_,weight= tasks[i]
            if type_ == 'l':
                preds = tf.nn.sigmoid(logits)
                predictions.append(preds)
                probablities.append(preds)
            elif type_ == 's':
                preds = tf.argmax(logits)
                probs = tf.nn.softmax(logits)
                predictions.append(preds)
                probablities.append(probs)


        if training:
            with tf.variable_scope('training',reuse = False):
                init_lr,decay_steps,decay_rate = lr_schedule
                costs          = []  
                global_steps   = []
                learning_rates = []
                trainOps       = []
                for i,logits in enumerate(logits_splited):
                    target_num,task_type,weight = tasks[i]
                    if task_type == 's':
                        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,
                                                                              labels = targets[i],
                                                                              name   = 'task%dcost'%(i+1))
                    elif task_type == 'l':
                        tgt = tf.one_hot(targets[i],depth =target_num,on_value=1.0,off_value=0.0)
                        tgt = tf.reduce_sum(tgt,axis=1)
                        cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits,
                                                                       labels = tgt,
                                                                       name   = 'task%dcost'%(i+1))
                    global_step = tf.Variable(initial_value=0.0,dtype=tf.float32,
                                              trainable = False,name = 'globalStepForTask%d'%(i+1))
                    learning_rate = tf.train.exponential_decay(learning_rate = init_lr,
                                                               global_step = global_step,
                                                               decay_steps = decay_steps,
                                                               decay_rate=decay_rate,
                                                               staircase=True,
                                                               name='learningRateForTask%d'%(i+1))
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name = 'optimizer%d'%(i+1))
                    grads, variables = zip(*optimizer.compute_gradients(cost))
                    grads, global_norm = tf.clip_by_global_norm(grads,10)
                    trainOp = optimizer.apply_gradients(zip(grads, variables),global_step=global_step,
                                                        name = 'trainOpTask%d'%(i+1))
                    costs.append(tf.reduce_mean(cost))
                    global_steps.append(global_step)
                    learning_rates.append(learning_rate)
                    trainOps.append(trainOp)
                
                total_loss = 0
                for i,(_,_,weight) in enumerate(tasks):
                    total_loss += weight * costs[i]
                    
                global_step = tf.Variable(initial_value=0.0,dtype=tf.float32,
                                          trainable = False,name = 'globalStepForWholeTask')
                learning_rate = tf.train.exponential_decay(learning_rate = init_lr,
                                                           global_step = global_step,
                                                           decay_steps = decay_steps,
                                                           decay_rate=decay_rate,
                                                           staircase=True,
                                                           name='learningRateForWholeTask')
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name = 'optimizerWhole')
                grads, variables = zip(*optimizer.compute_gradients(total_loss))
                grads, global_norm = tf.clip_by_global_norm(grads,10)
                trainOp = optimizer.apply_gradients(zip(grads, variables),global_step=global_step,
                                                    name = 'trainOpWholeTask')
                costs.append(tf.reduce_mean(total_loss))
                global_steps.append(global_step)
                learning_rates.append(learning_rate)
                trainOps.append(trainOp)
                
                self.global_steps   = global_steps
                self.learning_rates = learning_rates
                self.trainops       = trainOps
                self.inputs         = [sentences,lengths]
                self.targets        = targets
                self.costs          = costs
                self.predict        = predictions
                self.probablity     = probablities
        
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
    def zip_data(self,data,target_indices):
        pass



if __name__=='__main__':
    import numpy as np
    tf.reset_default_graph()
    model = XNET(vocab      = [2200,32],
                 tasks      = [[2,'s',1.0],[4,'s',1.0],[470,'l',1.0]],
                 lstm_units = [256,256,256],
                 time_step  = 100,
                 batch_size = 72)
    print(model.trainops)
#    sess = tf.Session()
#    sess.run(tf.global_variables_initializer())
#    inputs  = np.random.randint(low = 0, high = 2200 , size = [30,100]).astype(np.int32)
#    targets = np.random.randint(low = 0, high = 2 , size = [30]).astype(np.int32)
#    lengths = np.random.randint(low=10 , high = 100 , size = [30]).astype(np.int32)
#    
#    cost = sess.run(model.costs[0] , feed_dict = {model.inputs[0]:inputs,
#                                                  model.inputs[1]:lengths,
#                                                  model.targets[0]:targets})
    
    
    
    