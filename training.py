# -*- coding: utf-8 -*-

import tensorflow as tf
import DataFeeder
from PreTainRL import PreTrainModel

tf.reset_default_graph()
vocab = DataFeeder.vocab.vectors



trainner = PreTrainModel(vocab      = vocab,
                         lstm_units = [128,256,128],
                         target_cls = 1400,
                         poome      = 'max',
                         training   = True,
                         keep_prob  = .5)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

DF = DataFeeder.DataFeeder('PairedCorpus.test.txt')

results = []

for i in range(10):
    data = DF.next_batch_pair_task(10)
    feed_in = dict(zip(trainner.rl_placeholders,data))
    sess.run(trainner.rl_trainOp,feed_dict=feed_in)
    x = sess.run(trainner.check,feed_dict=feed_in)
    results.append(x)
    if i%1==0:
        cost = sess.run(trainner.rl_cost,feed_dict=feed_in)
        print(i,cost)

