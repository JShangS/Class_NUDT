# coding:utf-8

import tensorflow as tf

g = tf.Graph()
with g.as_default():
    a = tf.constant(2, name='a')
    print(a.name, a)
    b = tf.constant(3, name='b')
    print(b.name, b)
    # 返回 b:0    其中b:0 表示名字为b的第0个；
    # Tensor("b:0", shape=(), dtype=int32)
    # 表示 tensor的三个特征，名字，形状，数据类型
    x = tf.add(a, b, name='add')
    print(x.name, x)

# 这里和上一个案例的区别在于自定义了一个计算图，上一个案例使用的是默认的计算图
# 可以通过这种方式定义多个计算图