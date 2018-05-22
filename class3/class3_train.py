# 该模型实现如下的功能，输入两个MINIST图片，判断是不是同一个数字。
# 输入  负样本对：X1=6的图片 ， X2=9的图片 输出：1

# 输入  正样本对：X1=3的图片 ， X2=3的图片 输出：0

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


##############平衡样本#############
def balanced_batch(batch_x, batch_y, num_cls):
    batch_size = len(batch_y)
    pos_per_cls_e = round(batch_size / 2 / num_cls)

    index = batch_y.argsort()
    ys_1 = batch_y[index]
    #print(ys_1)

    num_class = []
    pos_samples = []
    neg_samples = set()
    cur_ind = 0
    for item in set(ys_1):
        num_class.append((ys_1 == item).sum())
        num_pos = pos_per_cls_e
        while (num_pos > num_class[-1]):
            num_pos -= 2
        pos_samples.extend(
            np.random.choice(
                index[cur_ind:cur_ind + num_class[-1]], num_pos,
                replace=False).tolist())
        neg_samples = neg_samples | (set(
            index[cur_ind:cur_ind + num_class[-1]]) - set(list(pos_samples)))
        cur_ind += num_class[-1]

    neg_samples = list(neg_samples)

    x1_index = pos_samples[::2]
    x2_index = pos_samples[1:len(pos_samples) + 1:2]

    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples) + 1:2])

    p_index = np.random.permutation(len(x1_index))
    x1_index = np.array(x1_index)[p_index]
    x2_index = np.array(x2_index)[p_index]

    r_x1_batch = batch_x[x1_index]
    r_x2_batch = batch_x[x2_index]
    r_y_batch = np.array(
        batch_y[x1_index] != batch_y[x2_index], dtype=np.float)
    r_y_batch = np.expand_dims(r_y_batch, 0)
    return r_x1_batch, r_x2_batch, r_y_batch


###########计算准确率################################
def calc_accuracy(pred, y):
    accuracy = tf.equal(tf.to_float(pred > 2.5), y)
    # accuracy = tf.equal(pred, y)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy, name='Accuracy')
    return sess.run(accuracy, feed_dict={x1: xs_t1, x2: xs_t2, y: y_ts})


##################添加网络的函数2个输入2输出###########################
def add_layer(inputs1,
              inputs2,
              in_size,
              out_size,
              Wname,
              Bname,
              activation_function=None):
    Weights = tf.Variable(
        tf.random_normal([in_size, out_size]),
        name=Wname)  #random_normal#zeros
    biases = tf.Variable(tf.random_normal([1, out_size]), name=Bname)
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
# train_data = mnist.train.images  # Returns np.array
# train_labels = np.asarray(mnist.train.labels, dtype=np.float)
# test_data = mnist.test.images
# test_labels = np.asarray(mnist.train.labels, dtype=np.float)

#####################构图#####################
batch_size = 64
x1 = tf.placeholder(tf.float32, shape=[None, 784], name='input1')
x2 = tf.placeholder(tf.float32, shape=[None, 784], name='input2')
y = tf.placeholder(tf.float32, shape=[1, None], name='label')
###第一层###################
L11, L12 = add_layer(x1, x2, 784, 500, 'W1', 'B1',
                     tf.nn.leaky_relu)  #tf.nn.sigmoid
###第二个层###################
L21, L22 = add_layer(
    L11, L12, 500, 10, 'W2', 'B2',
    tf.nn.leaky_relu)  #tf.nn.softmax #tf.nn.relu#tf.nn.leaky_relu
###############计算Ew和预测##################
Ew = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(L21, L22)), 1), name='Ew')
pred = Ew  #tf.nn.sigmoid(Ew)
# pred = tf.to_float(tf.equal(tf.argmax(L21, 1), tf.argmax(L22, 1)))
############loss#############
Q = tf.constant(5, dtype=tf.float32)
# loss = tf.reduce_mean(tf.square(y - pred))
loss = tf.add(
    tf.multiply(
        tf.multiply(tf.subtract(1.0, y), tf.div(2.0, Q)), tf.square(Ew)),
    tf.multiply(
        tf.multiply(tf.multiply(2.0, y), Q),
        tf.exp(tf.multiply(tf.div(-2.77, Q), Ew))))
lr = 1e-2
optimizer = tf.train.AdamOptimizer(
    learning_rate=lr)  #AdamOptimizer#GradientDescentOptimizer
train = optimizer.minimize(loss=(loss))  #tf.reduce_mean
################会话############################
ACC = []
saver = tf.train.Saver()
with tf.Session() as sess:
    iter = 20001
    sess.run(tf.global_variables_initializer())
    for itera in range(iter):
        batch_x, batch_y = mnist.train.next_batch(batch_size * 2)
        xs_1, xs_2, y_s = balanced_batch(batch_x, batch_y, num_cls=10)
        _, l, y_pred, rEw = sess.run(
            [train, loss, pred, Ew], feed_dict={
                x1: xs_1,
                x2: xs_2,
                y: y_s
            })
        if itera % 1000 == 0:
            print('\n')
            print('第', itera, '次迭代，损失为： ', sess.run(tf.reduce_mean(l)))
            print(y_s.reshape((1, -1)))
            print('Ew', rEw, '\n')
            #######################测试#############################
            batch_x, batch_y = mnist.test.next_batch(batch_size * 2)
            xs_t1, xs_t2, y_ts = balanced_batch(batch_x, batch_y, num_cls=10)
            #######准确率########################
            # ACC.append(calc_accuracy(pred, y))
            ACC = calc_accuracy(pred, y)
            print('准确率 ACC =', ACC)
            saver.save(sess, './model_class3/my_test_model')

tf.logging.set_verbosity(old_v)