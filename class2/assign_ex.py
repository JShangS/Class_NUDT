import tensorflow as tf
import numpy as np

b=tf.Variable(initial_value=np.random.randint(10), name='b')
#assign_op=tf.add(b,1)
assign_op=tf.assign(b,b+1)
out=assign_op*2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3):
        print(sess.run([out,b]))





