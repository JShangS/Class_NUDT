# coding:utf-8

import tensorflow as tf

a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
c = tf.add(a, b, name='c')

with tf.Session() as sess:
    print(sess.run(c))
    # 除了可以通过sess.run 获得tensor的值之外，还可以通过tensor的eval方法获得
    print(c.eval())