import tensorflow as tf
import numpy as np

X=tf.placeholder(tf.float32)

def model(X):
    w=tf.Variable(name="w", initial_value=tf.random_normal(shape=[1])) 
    m=tf.multiply(X,w)
    return m

def train_graph(X):
    m=model(X)
    a=tf.add(m,X)

    return a

def test_graph(X):
    m=model(X)
    b=tf.add(m,X)

    return b

a=train_graph(X)
b=test_graph(X)



#def model(X,reuse_flag):
#    with tf.variable_scope('model',reuse=reuse_flag):
#        w=tf.get_variable(name="w", initializer=tf.random_normal(shape=[1])) 
#        #w=tf.Variable(name="w", initial_value=tf.random_normal(shape=[1])) 
#    m=tf.multiply(X,w)
#    return m

#def train_graph(X,reuse_flag=False):
#    m=model(X,reuse_flag)
#    a=tf.add(m,X)

#    return a

#def test_graph(X,reuse_flag=False):
#    m=model(X,reuse_flag)
#    b=tf.add(m,X)

#    return b

#a=train_graph(X)
#b=test_graph(X,reuse_flag=True)

X_in=1.2

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    ar=sess.run(a, feed_dict={X:X_in})
    br=sess.run(b, feed_dict={X:X_in})
    print("ar=",ar)
    print("br=",br)
