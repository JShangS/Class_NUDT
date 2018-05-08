import tensorflow as tf
import numpy as np

b=tf.Variable(initial_value=np.random.randint(10),name='b')
b=tf.add(b,1)
out=b*2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        print(sess.run([out,b]))


