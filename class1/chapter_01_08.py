# coding:utf-8

import tensorflow as tf

a = tf.placeholder(tf.int16)
x = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

mul = tf.multiply(a, x) # 将a,x 相乘
y = tf.add(mul, b)

# session 里面没有传入计算图的时候使用默认计算图
with tf.Session() as sess:
    # 通常用字典数据结构作为feed 机制的输入
    print('multiply with variable: %i' % sess.run(mul, feed_dict={a: 2, x: 3}))
    print('addtion with variable: %i' % sess.run(y, feed_dict={a: 2, x: 3, b: 3}))


