# coding:utf-8

import tensorflow as tf
import numpy as np

aa = np.random.rand(1)
for i in range(5):
    print(aa)
# 每次都会是一样的结果，因为aa已经赋值了，并且只被赋值一次

a = tf.random_normal([1], name='random')
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(a))
        # 每次的结果都不会一样，因为每次都是重新去运行该a节点