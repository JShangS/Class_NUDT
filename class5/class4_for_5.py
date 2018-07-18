##将class4中的训练过程中的每层的W，b的变化，ACC的变化记录下来，等参数变化记录下来
##在tensorboard 中显示

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from class_lenet import Lennet
LOGDIR = 'F:/PycharmProjects/Class_NUDT/class5/tensorboard/'


def calc_accuracy(pred, y):
    with tf.name_scope('ACC') as scope:
        accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.cast(accuracy, tf.float32)
        accuracy = tf.reduce_mean(accuracy, name='Accuracy')
        ACC = sess.run(
            accuracy, feed_dict={
                lennet.input_x: x_test,
                lennet.input_y: y_test
            })
        tf.summary.histogram('accuracy', ACC)
    return ACC


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
pred = tf.nn.softmax(lennet.logist, name='pred')
# print(pred.shape)
loss = tf.reduce_mean(
    -tf.reduce_sum(tf.multiply(tf.log(pred), lennet.input_y)),
    name='loss')  #cross_entropy
# loss = tf.reduce_mean(tf.square(lennet.input_y - pred))
step = 1e-4
optimizer = tf.train.AdamOptimizer(
    step)  #AdamOptimizer#GradientDescentOptimizer
train = optimizer.minimize(loss=loss, name='train')
saver = tf.train.Saver()
##记录##################################################
merge_all = tf.summary.merge_all()
writer = tf.summary.FileWriter(LOGDIR + 'JS_board2S/')
#######################################################
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iter = 20001
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
        ##############每隔50次记录一次####################
        if itera % 50 == 0:
            s = sess.run(
                merge_all,
                feed_dict={
                    lennet.input_x: x_train,
                    lennet.input_y: y_train
                })
            writer.add_summary(s, itera)
            writer.flush()
        ################################################
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
            saver.save(
                sess,
                'F:/PycharmProjects/Class_NUDT/model_class5/my_test_model')
    writer.close()
tf.logging.set_verbosity(old_v)