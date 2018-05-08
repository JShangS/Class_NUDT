import tensorflow as tf
import numpy as np

np.random.seed(1)
b=tf.Variable(initial_value=np.random.randint(10),name='b')
a=tf.Variable(initial_value=np.random.randint(10),name='a')
print(b)
print('b without init',b)
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    sess.run(tf.variables_initializer([a]))
    

    ra=sess.run([a])
    rb=sess.run([b])
    print('b with init',rb)