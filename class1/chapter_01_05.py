# coding:utf-8

import tensorflow as tf

# 定义两个常量向量
a = tf.constant([1. , 2.], name='a')
b = tf.constant([2. , 3.], name='b')
result = a + b
print(result)

# 第一种使用session的方式，明确的调用和关闭会话资源
sess = tf.Session()
xxx = sess.run(result)
print(xxx)
sess.close()

# 第二种使用session的方式， 通过python的上下文管理器自动关闭会话资源，推荐使用这种方式
with tf.Session() as sess:
    xxx = sess.run(result)
    print(xxx)
