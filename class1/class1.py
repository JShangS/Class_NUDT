## 输入：A,B,C

## 其中A是3x4 float32 随机矩阵，B是4x3 float32 随机矩阵， C是3x3 float32 随机矩阵

## 且A,B,C 各不相等

## 用tensorflow实现：AxB+C 并打印出A, B, C及（AxB+C）的数值结果

import tensorflow as tf
import numpy as np


A = tf.placeholder(tf.float32, shape=(3, 4))
B = tf.placeholder(tf.float32, shape=(4, 3))
C = tf.placeholder(tf.float32, shape=(3, 3))

# np.random.seed(1)
An = np.random.randn(3, 4)
Bn = np.random.randn(4, 3)
Cn = np.random.randn(3, 3)

D = tf.add(tf.matmul(A, B), C)

with tf.Session() as sess:
    PA, PB, PC, PD = sess.run([A, B, C, D], feed_dict={A: An, B: Bn, C: Cn})
    print('PA=\n', PA, '\n', 'PB=\n', PB, '\n', 'PC=\n', PC, '\n', 'PD=\n', PD, '\n')




