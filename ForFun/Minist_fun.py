import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, active_funtion=None):
    Weights = tf.Variable(initial_value=tf.zeros([in_size, out_size]))
    Biases = tf.Variable(initial_value=tf.zeros([1, out_size]))
    WB = tf.matmul(inputs, Weights) + Biases
    if active_funtion is None:
        outputs = WB
    else:
        outputs = active_funtion(WB)
    return outputs


def cal_accuracy(pred, y):
    accuracy = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return sess.run(accuracy, feed_dict={x: x_test, y: y_test})


##########读取数据################################
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
#####################构图#####################
x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
y = tf.placeholder(tf.float32, shape=[None, 10], name='label')
#################加网络,1层########################
L1 = add_layer(x, 784, 10)
#####################################
pred = tf.nn.softmax(L1)
# loss = -tf.reduce_sum(y * tf.log(pred))
loss = tf.sqrt(tf.reduce_sum(tf.square(pred - y),1))
step = 0.01################step很重要，0.1就不行,要小
optimizer = tf.train.GradientDescentOptimizer(step)
train = optimizer.minimize(loss)

itera = 20000
batch_size = 64
#############会话#################
Loss = []
acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iterai in range(itera):
        x_train, y_train = mnist.train.next_batch(batch_size)
        # print(x_train.shape, y_train.shape)
        _, l = sess.run([train, loss], feed_dict={x: x_train, y: y_train})
        Loss = np.append(Loss,l)
        if iterai % 1000 == 0:
            ll = sess.run(tf.reduce_mean(Loss))
            print('损失: ', ll)
            x_test, y_test = mnist.test.next_batch(batch_size)
            acc = np.append(acc,cal_accuracy(pred, y))
            ACC = sess.run(tf.reduce_mean(acc))
            print('准确率：', ACC)
