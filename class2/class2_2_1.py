# 二. 分类机器学习：

# 采用逻辑斯蒂回归（如课程中例子）建立二维数据二分类模型。
# 两类数据服从二维高斯分布：
# 类A：(??,??)~??(3, 6, 1, 1, 0)
# 类B：(??,??)~??(6, 3, 1, 1, 0)

# 1）分别为类A，类B各随机生成100个样本作为训练数据train_data，30个样本作为验证数据validation_data，30个样本作为测试数据test_data。
# 3）建立逻辑斯蒂回归模型并用GradientDescentOptimizer优化器（参数默认0.01）进行优化学习。
# 4）采用参数复用方式构建学习、验证两个计算路径。学习过程中，进行500次迭代，每次按顺序取train_data中的20个数据进行训练。每100次迭代用30个验证样本validation_data进行验证。训练过程中，打印训练的损失函数值及模型在验证集上的精度。
# 5) 用ckpt方式保存模型，每100次迭代保存一次。
# 6）在另一个py文件中写测试流程，用import_meta_graph导入计算图，get_tensor_by_name得到输入placeholder，以及ACC的tensor，建立测试流程，并用test_data对模型进行测试，输出ACC。
# 7)  将分类结果绘图（例如用matplotlib），（如课程中例子）A、B类测试数据(分别在图中用‘+’以及‘o’表示)，分类模型以分割线表示。

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def calc_accuracy(logist, y):
    accuracy = tf.equal(tf.argmax(logist, 1), tf.argmax(y, 1))
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return sess.run(accuracy, feed_dict={x: X_test, y: Y_test})


##############基本设置########################
mu1x = 6
mu2x = 3
mu1y = 3
mu2y = 6
sigma = 1
#训练设置#####################################
step = 0.01
batch_train = 20
len_train = 100
#训练数据
Ax = sigma * np.random.randn(len_train, 1) + mu1x
Ay = sigma * np.random.randn(len_train, 1) + mu1y
Bx = sigma * np.random.randn(len_train, 1) + mu2x
By = sigma * np.random.randn(len_train, 1) + mu2y

# plt.plot(Ax,Ay,'r+')
# plt.plot(Bx,By,'bo')
# plt.show()

XA = np.append(Ax, Ay, axis=1)
XB = np.append(Bx, By, axis=1)
X = np.append(XA, XB, axis=0)
# for i in range(100):
#     print(X[i,:])
Y0 = np.zeros(shape=(len_train, 1))
Y01 = np.append(Y0, np.ones(shape=(len_train, 1)), axis=1)
Y1 = np.ones(shape=(len_train, 1))
Y10 = np.append(Y1, np.zeros(shape=(len_train, 1)), axis=1)
Y = np.append(Y10, Y01, axis=0)
# print(Y[0:20])
# print(Y)
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

####构图########################################
W = tf.Variable(initial_value=tf.zeros([2, 2]), name='Weigh', dtype=tf.float32)
b = tf.Variable(initial_value=tf.zeros([2]), name='bias', dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=(None, 2), name='x')
y = tf.placeholder(tf.float32, shape=(None, 2), name='y')

logist = tf.matmul(x, W) + b
pred = tf.sigmoid(logist)
# pred = tf.matmul(x, W) + b
# 定义损失函数
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logist, labels=y))
loss = tf.reduce_mean(tf.square(y - pred))
optimizer = tf.train.GradientDescentOptimizer(step)
train = optimizer.minimize(loss=loss)
acc = []
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(500):
        avloss = 0.
        ###################训练##############################
        total_batch = int(len_train * 2 / batch_train)
        for i in range(int(total_batch)):
            #    print(i)
            train_X = X[i * batch_train:(i + 1) * batch_train, :]
            train_Y = Y[i * batch_train:(i + 1) * batch_train, :]
            #    print(train_Y)
            _, l = sess.run([train, loss], feed_dict={x: train_X, y: train_Y})
            avloss += l / total_batch
            ###################测试##############################
            if j % 100 == 0 and i == 0:
                saver.save(sess, './model/my_test_model')
                print('\n*************第', j, '次迭代*****************', '\n')
                print('损失: ', avloss, '\n')
                rw = sess.run(W)
                rb = sess.run(b)
                print('训练结果:\n')
                print('W=', rw, '\n\n', 'b=', rb, '\n')
                acc.append(calc_accuracy(logist, y))
                print('准确率 ACC =', acc[-1])
# ###################画图##############################
#     N = 100
#     x1_min, x1_max =XA_test.min()-1, XA_test.max()+1   # x的范围
#     x2_min, x2_max = XB_test.min()-1, XB_test.max()+1   # y的范围
#     t1 = np.linspace(x1_min, x1_max, N)
#     t2 = np.linspace(x2_min, x2_max, N)
#     x1, x2 = np.meshgrid(t1, t2)                    # 生成网格采样点
#     x1.shape = (-1,1)
#     x2.shape = (-1,1)
#     X_feed = np.append(x1, x2, axis=1)
#     y_pred = sess.run(tf.argmax(logist, 1) , feed_dict={x: X_feed})
#     ###############重新给形状########################
#     x1, x2 = np.meshgrid(t1, t2)
#     y_pred = y_pred.reshape(x1.shape)
#     # print(y_pred)
#     plt.plot(Ax_test,Ay_test, 'k+')
#     plt.plot(Bx_test,By_test, 'bo')
#     cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080'])
#     plt.pcolormesh(x1, x2, y_pred, cmap=cm_light)     # 预测值的显示
#     plt.show()
