import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from class_lenet import Lennet


def calc_accuracy(pred, y):
    accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return sess.run(
        accuracy, feed_dict={
            lennet.input_x: x_test,
            lennet.input_y: y_test
        })


old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#####################构图#####################
batch_size = 64
# x = tf.placeholder(tf.float32, shape=[None, 1024], name='input')
# x = tf.reshape(x, [-1, 32, 32, 1])  #
# y = tf.placeholder(tf.float32, shape=[None, 10], name='label')
#############loss#################
lennet = Lennet(10)
pred = tf.nn.softmax(lennet.logist)
# loss = -tf.reduce_mean(
#     tf.reduce_sum(pred * tf.log(lennet.input_y),
#                   reduction_indices=[1]))  #cross_entropy
loss = tf.reduce_mean(tf.square(lennet.input_y - pred))
step = 1e-4
optimizer = tf.train.AdamOptimizer(
    step)  #AdamOptimizer#GradientDescentOptimizer
train = optimizer.minimize(loss=loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iter = 10001
    acc = []
    for itera in range(iter):
        x_train, y_train = mnist.train.next_batch(batch_size)
        x_train = np.reshape(x_train, [-1, 28, 28, 1])  #
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        [_, r_pred, r_loss] = sess.run(
            [train, pred, loss],
            feed_dict={
                lennet.input_x: x_train,
                lennet.input_y: y_train
            })
        if itera % 1000 == 0 and itera > 1:
            x_test, y_test = mnist.test.next_batch(batch_size)
            x_test = np.reshape(x_test, [-1, 28, 28, 1])  #
            x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)),
                            'constant')
            # acc = np.append(acc, calc_accuracy(pred, lennet.input_y))
            # ACC = sess.run(tf.reduce_mean(acc))
            ACC = calc_accuracy(pred, lennet.input_y)
            print('第', itera, '次迭代')
            print('损失是：', r_loss)
            print('准确率: ', ACC)

tf.logging.set_verbosity(old_v)