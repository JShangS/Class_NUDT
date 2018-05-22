import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def calc_accuracy(pred, y):
    accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return restore_sess.run(
        accuracy, feed_dict={
            input_x: x_test,
            input_y: y_test
        })


##########读取数据################################
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#######################恢复图##############################
batch_size = 64
tf.reset_default_graph()
restore_graph = tf.Graph()
with tf.Session(graph=restore_graph) as restore_sess:
    restore_saver = tf.train.import_meta_graph(
        './model_class4/my_test_model.meta')
    restore_saver.restore(restore_sess,
                          tf.train.latest_checkpoint('./model_class4'))
    graph = tf.get_default_graph()
    input_x = graph.get_tensor_by_name('lenet_var/input:0')
    input_y = graph.get_tensor_by_name('lenet_var/label:0')
    pred = graph.get_tensor_by_name('pred:0')
    loss = graph.get_tensor_by_name('loss:0')
    #######################测试#############################
    ACC = []
    l = []
    numitera = 100
    for i in range(numitera):
        if (i+1) % (numitera/10) == 0:
            print((i+1)/numitera*100, '%')
        x_test, y_test = mnist.train.next_batch(batch_size)
        x_test = np.reshape(x_test, [-1, 28, 28, 1])  #
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        ACC = np.append(ACC, calc_accuracy(pred, input_y))
        l = np.append(l,
                      restore_sess.run(
                          loss, feed_dict={
                              input_x: x_test,
                              input_y: y_test
                          }))
    print('损失 Loss =', restore_sess.run(tf.reduce_mean(l)))
    print('准确率 ACC =', restore_sess.run(tf.reduce_mean(ACC)))

tf.logging.set_verbosity(old_v)
