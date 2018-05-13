import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


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
    Ax_test = sigma*np.random.randn(len_test, 1) + mu1x
    Ay_test = sigma*np.random.randn(len_test, 1) + mu1y
    Bx_test = sigma*np.random.randn(len_test, 1) + mu2x
    By_test = sigma*np.random.randn(len_test, 1) + mu2y

    XA_test = np.append(Ax_test,Ay_test, axis=1)
    XB_test = np.append(Bx_test,By_test, axis=1)
    X_test = np.append(XA_test,XB_test, axis=0)

    Y0_test = np.zeros(shape=(len_test, 1))
    Y1_test = np.ones(shape=(len_test, 1))
    Y_test = np.append(Y0_test,Y1_test, axis=0)

with tf.Session(graph=restore_graph) as restore_sess:
    restore_saver = tf.train.import_meta_graph('./model/my_test_model.meta')
    restore_saver.restore(restore_sess, tf.train.latest_checkpoint('./model'))
    graph = tf.get_default_graph()
    x_test = graph.get_tensor_by_name('x_validate:0')
    y_test = graph.get_tensor_by_name('y_validate:0')
    W = graph.get_tensor_by_name('Weigh:0')
    b = graph.get_tensor_by_name('bias:0')
    accuracy_test = graph.get_tensor_by_name('Accuracy:0')
    accuracy = restore_sess.run(accuracy_test, feed_dict={x_test:X_test,y_test:Y_test})
    print('\n准确率 ACC =', accuracy)
    ###################画图##############################
    N = 100
    x1_min, x1_max =XA_test.min()-1, XA_test.max()+1   # x的范围
    x2_min, x2_max = XB_test.min()-1, XB_test.max()+1   # y的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, N)
    x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
    x1.shape = (-1, 1)
    x2.shape = (-1, 1)
    # 将网格上的每个点输入网络得到Pred值，根据Pred的值确定图块的颜色
    X_feed = np.append(x1, x2, axis=1)
    pred = graph.get_tensor_by_name('Pred:0')
    y_pred = restore_sess.run(pred, feed_dict={x_test: X_feed})
    isB = np.greater(y_pred, 0.5)
    ###############重新给形状########################
    x1, x2 = np.meshgrid(t1, t2)
    isB = isB.reshape(x1.shape)
    plt.plot(Ax_test,Ay_test, 'k+')
    plt.plot(Bx_test,By_test, 'bo')
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
    plt.pcolormesh(x1, x2, isB, cmap=cm_light)     # 预测值的显示
    plt.show()

