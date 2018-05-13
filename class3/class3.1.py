import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


##################添加网络的函数###########################
def add_layer(inputs1, inputs2, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.zeros([in_size, out_size]))#random_normal#
    biases = tf.Variable(tf.zeros([1, out_size]))
    Wx_plus_b1 = tf.matmul(inputs1, Weights) + biases
    Wx_plus_b2 = tf.matmul(inputs2, Weights) + biases
    if activation_function is None:
        outputs1 = Wx_plus_b1
        outputs2 = Wx_plus_b2
    else:
        outputs1 = activation_function(Wx_plus_b1)
        outputs2 = activation_function(Wx_plus_b2)
    return outputs1, outputs2


##########读取数据################################
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('./data/mnist', one_hot=False)
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.float)
test_data = mnist.test.images
test_labels = np.asarray(mnist.train.labels, dtype=np.float)

#####################构图#####################
x1 = tf.placeholder(tf.float32, shape=[None, 784], name='input1')
x2 = tf.placeholder(tf.float32, shape=[None, 784], name='input2')
y = tf.placeholder(tf.float32, shape=[None, 1], name='label')
###第一层###################
L11, L12 = add_layer(x1, x2, 784, 300, tf.nn.relu)
###第二个层###################
L21, L22 = add_layer(L11, L12, 300, 10, tf.nn.relu)
# ###第3个层###################
# L31, L32 = add_layer(L21, L22, 200, 10)
############loss#############
Q = 5
Ew = tf.sqrt(tf.reduce_sum(tf.square(L21 - L22), 1))
loss = (1 - y) * 2 / Q * Ew * Ew + y * 2 * Q * tf.exp(-2.77 / Q * Ew)
step = 0.1
optimizer = tf.train.AdamOptimizer(
    step)  #AdamOptimizer,GradientDescentOptimizer
train = optimizer.minimize(loss=loss)
################会话############################
batch_size = 64
with tf.Session() as sess:
    iter = 10000
    sess.run(tf.global_variables_initializer())
    for itera in range(iter):
        x_1, y_1 = mnist.train.next_batch(batch_size)
        x_2, y_2 = mnist.train.next_batch(batch_size)
        y_s = np.array(y_1 != y_2, dtype=np.float).reshape(-1, 1)
        _, l = sess.run([train, loss], feed_dict={x1: x_1, x2: x_2, y: y_s})
        if itera % 1000 == 0:
            print(itera, ': ', sess.run(tf.reduce_mean(l)), '\n')

tf.logging.set_verbosity(old_v)
