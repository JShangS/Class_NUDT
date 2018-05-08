# coding:utf-8

import tensorflow as tf

def graph_a():
    # 定义两个常量向量
    a = tf.constant([1. , 2.], name='a')
    b = tf.constant([2. , 3.], name='b')
    result = a + b
    return result

# 创建一个会话
sess = tf.Session()
# 通过会话获得计算结果
# xxx = sess.run(result)
# 这里会报错，因为result并不是一个全局变量而是函数内部的局部变量
# print(xxx)



# 需要按照如下方式运行
result = graph_a()
xxx = sess.run(result)
print(xxx)