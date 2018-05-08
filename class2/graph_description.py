import tensorflow as tf
import numpy as np

x=tf.placeholder(tf.int32,shape=(1,2),name='input_data')
w=tf.Variable(np.random.randint(10,size=(2,1)),name='col_vector')

b=tf.Variable(initial_value=np.random.randint(10),name='bias')
c=tf.Variable(np.random.randint(10),name='scalar')

rMatMul=tf.matmul(x,w)
#rMatMul=x*w
rAdd=tf.add(rMatMul,b)

rMul=tf.multiply(rAdd,c)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([w,b,c,rMul,rAdd, rMatMul], feed_dict={x:[[25,3]]}))


