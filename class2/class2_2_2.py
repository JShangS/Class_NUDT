import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def calc_accuracy(logist, y):
    accuracy = tf.equal(tf.argmax(logist, 1), tf.argmax(y, 1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return restore_sess.run(accuracy, feed_dict={x: X_test, y: Y_test})


#######################恢复图##############################
tf.reset_default_graph()
restore_graph = tf.Graph()
with restore_graph.as_default():
    ##############基本设置########################
    mu1x = 6
    mu2x = 3
    mu1y = 3
    mu2y = 6
    sigma = 1
    ####测试####################################
    #测试设置
    batch_test = 30
    len_test = 30
    #测试数据
    Ax_test = sigma * np.random.randn(len_test, 1) + mu1x
    Ay_test = sigma * np.random.randn(len_test, 1) + mu1y
    Bx_test = sigma * np.random.randn(len_test, 1) + mu2x
    By_test = sigma * np.random.randn(len_test, 1) + mu2y

    XA_test = np.append(Ax_test, Ay_test, axis=1)
    XB_test = np.append(Bx_test, By_test, axis=1)
    X_test = np.append(XA_test, XB_test, axis=0)

    Y0_test = np.zeros(shape=(len_test, 1))
    Y01_test = np.append(Y0_test, np.ones(shape=(len_test, 1)), axis=1)
    Y1_test = np.ones(shape=(len_test, 1))
    Y10_test = np.append(Y1_test, np.zeros(shape=(len_test, 1)), axis=1)
    Y_test = np.append(Y10_test, Y01_test, axis=0)

with tf.Session(graph=restore_graph) as restore_sess:
    restore_saver = tf.train.import_meta_graph('./model/my_test_model.meta')
    restore_saver.restore(restore_sess, tf.train.latest_checkpoint('./model'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    W = graph.get_tensor_by_name('Weigh:0')
    b = graph.get_tensor_by_name('bias:0')
    logist = tf.matmul(x, W) + b
    acc = calc_accuracy(logist, y)
    print('\n准确率 ACC =', acc)
    ###################画图##############################
    N = 500
    x1_min, x1_max = XA_test.min() - 1, XA_test.max() + 1  # x的范围
    x2_min, x2_max = XB_test.min() - 1, XB_test.max() + 1  # y的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, N)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    X_feed = np.append(x1, x2, axis=1)
    y_pred = restore_sess.run(tf.argmax(logist, 1), feed_dict={x: X_feed})
    ###############重新给形状########################
    x1, x2 = np.meshgrid(t1, t2)
    y_pred = y_pred.reshape(x1.shape)
    # print(y_pred)
    plt.plot(Ax_test, Ay_test, 'k+')
    plt.plot(Bx_test, By_test, 'bo')
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
    plt.pcolormesh(x1, x2, y_pred, cmap=cm_light)  # 预测值的显示
    plt.show()
##################################另一种恢复#######################
# sess =tf.Session()
# saver = tf.train.import_meta_graph('./model/my_test_model.meta')
# saver.restore(sess, tf.train.latest_checkpoint('./model'))
# graph = tf.get_default_graph()
# x = graph.get_tensor_by_name('x:0')
# y = graph.get_tensor_by_name('y:0')
# logist = graph.get_tensor_by_name('add:0')
# accuracy = tf.equal(tf.argmax(logist, 1), tf.argmax(y, 1))
# accuracy = tf.cast(accuracy, tf.float32)
# accuracy = tf.reduce_mean(accuracy)
# acc = sess.run(accuracy, feed_dict={x: X_test, y: Y_test})
# print('准确率：', acc)
