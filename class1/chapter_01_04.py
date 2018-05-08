# coding:utf-8

import tensorflow as tf

# 定义两个常量向量
a = tf.constant([1. , 2.], name='a')
b = tf.constant([2. , 3.], name='b')
result = a + b
print(result)
# 只会得到未运行的tensor

# 需要按照如下方式运行，创建一个会话并在会话中执行默认计算图
sess = tf.Session()
xxx = sess.run(result)
print(xxx)
