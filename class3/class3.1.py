# 该模型实现如下的功能，输入两个MINIST图片，判断是不是同一个数字。
# 输入  负样本对：X1=6的图片 ， X2=9的图片 输出：1

# 输入  正样本对：X1=3的图片 ， X2=3的图片 输出：0

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


###########计算准确率################################
def calc_accuracy(pred, y):
    accuracy = tf.equal(tf.to_float(pred>0.5), y)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return sess.run(accuracy, feed_dict={x1: X_test1, x2: X_test2, y: Y_test})


##################添加网络的函数2个输入2输出###########################
def add_layer(inputs1, inputs2, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(
        tf.random_normal([in_size, out_size]),
        name='Weights')  #random_normal#zeros
    biases = tf.Variable(tf.random_normal([1, out_size]), name='biases')
    Wx_plus_b1 = tf.matmul(inputs1, Weights) + biases
    Wx_plus_b2 = tf.matmul(inputs2, Weights) + biases
    if activation_function is None:
        outputs1 = Wx_plus_b1
        outputs2 = Wx_plus_b2
    else:
        outputs1 = activation_function(Wx_plus_b1)
        outputs2 = activation_function(Wx_plus_b2)
    return outputs1, outputs2


##################添加网络的函数2个输入2输出###########################
def add_layer2(inputs1, inputs2, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.zeros([in_size * 2,
                                    out_size]))  #random_normal#zeros
    biases = tf.Variable(tf.zeros([1, out_size]))
    inputs = np.append(inputs1, inputs2, axis=1)
    Wx_plus_b = tf.matmul(inputdata, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs1, outputs2


##########读取数据################################
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('./data/mnist', one_hot=False)
# train_data = mnist.train.images  # Returns np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.float)
# test_data = mnist.test.images
# test_labels = np.asarray(mnist.train.labels, dtype=np.float)

#####################构图#####################
x1 = tf.placeholder(tf.float32, shape=[None, 784], name='input1')
x2 = tf.placeholder(tf.float32, shape=[None, 784], name='input2')
y = tf.placeholder(tf.float32, shape=[None, 1], name='label')
###第一层###################
L11, L12 = add_layer(x1, x2, 784, 500, tf.nn.relu)
###第二个层###################
L21, L22 = add_layer(L11, L12, 500, 10, tf.nn.relu)  #tf.nn.softmax #tf.nn.relu
###############计算Ew和预测##################
Ew = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(L21, L22)), 1))
pred = 2 * tf.nn.sigmoid(Ew) - 1
# pred = tf.to_float(tf.equal(tf.argmax(L21, 1), tf.argmax(L22, 1)))
############loss#############
Q = tf.constant(5, dtype=tf.float32)
# loss = tf.reduce_mean(tf.square(y - pred))
loss = tf.add(
    tf.multiply(
        tf.multiply(tf.subtract(1.0, y), tf.div(2.0, Q)), tf.square(Ew)),
    tf.multiply(
        tf.multiply(tf.multiply(y, 2.0), Q),
        tf.exp(tf.multiply(tf.div(-2.77, Q), Ew))))
lr = 0.001
optimizer = tf.train.AdamOptimizer(
    learning_rate=lr)  #AdamOptimizer#GradientDescentOptimizer
train = optimizer.minimize(loss=(loss))  #tf.reduce_mean
################会话############################
batch_size = 320
batch_size_test = 32
ACC = []
with tf.Session() as sess:
    iter = 200001
    sess.run(tf.global_variables_initializer())
    for itera in range(iter):
        y__s = []
        x__1 = []
        x__2 = []
        x_1, y_1 = mnist.train.next_batch(batch_size)
        x_2, y_2 = mnist.train.next_batch(batch_size)
        y_s = np.array(y_1 != y_2, dtype=np.float).reshape(-1, 1)
        #####降采样###################
        index0 = np.argwhere(y_s == 0)[:, 0]
        if len(index0) < 32:
            continue
        index1 = np.argwhere(y_s == 1)[:, 0]
        y__s = np.append(y_s[index0[0:32], :], y_s[index1[0:32], :], axis=0)
        x__1 = np.append(x_1[index0[0:32], :], x_1[index1[0:32], :], axis=0)
        x__2 = np.append(x_2[index0[0:32], :], x_2[index1[0:32], :], axis=0)
        _, l, y_pred, rEw = sess.run(
            [train, loss, pred, Ew], feed_dict={
                x1: x__1,
                x2: x__2,
                y: y__s
            })
        if itera % 1000 == 0:
            print('\n')
            print('第', itera, '次迭代，损失为： ', sess.run(tf.reduce_mean(l)))
            # print(y__s)
            # print(y_pred)
            # print(rEw)
            #######################测试#############################
            X_test1, Y_test1 = mnist.test.next_batch(batch_size_test)
            X_test2, Y_test2 = mnist.test.next_batch(batch_size_test)
            Y_test = np.array(
                Y_test1 != Y_test2, dtype=np.float).reshape(-1, 1)
            #######准确率########################
            ACC.append(calc_accuracy(pred, y))

            # print(np.shape(y__s), np.shape(x__1))
            # print(sess.run(pred, feed_dict={
            #     x1: X_test1,
            #     x2: X_test2,
            # }))
            # print('\n', Y_test)
            # print('准确率 ACC =', ACC[-1])
            # print(y_s)

tf.logging.set_verbosity(old_v)
