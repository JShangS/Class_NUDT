import tensorflow as tf  
import numpy as np


a=tf.placeholder(tf.int16)
b=tf.placeholder(tf.int16)

c=tf.add(a,b)

np.random.seed(1)
an=np.random.randint(10)
bn=np.random.randint(10)

with tf.Session() as sess:
    pa,pb, pc=sess.run([a,b,c], feed_dict={a:an, b:bn})
    print(pa,pb,pc)
