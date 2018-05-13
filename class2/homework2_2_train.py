import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
def calc_accuracy(logist, y):
    accuracy = tf.equal(logist,y)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return sess.run(accuracy)
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
Ax = sigma*np.random.randn(len_train, 1) + mu1x
Ay = sigma*np.random.randn(len_train, 1) + mu1y
Bx = sigma*np.random.randn(len_train, 1) + mu2x
By = sigma*np.random.randn(len_train, 1) + mu2y

XA = np.append(Ax,Ay, axis=1)
XB = np.append(Bx,By, axis=1)
X = np.append(XA,XB, axis=0)

Y0 = np.zeros(shape=(len_train, 1))
Y1 = np.ones(shape=(len_train, 1))
Y = np.append(Y0, Y1, axis=0)

####测试####################################
#验证设置
batch_validate = 30
len_validate = 30
#验证数据
Ax_validate = sigma*np.random.randn(len_validate, 1) + mu1x
Ay_validate = sigma*np.random.randn(len_validate, 1) + mu1y
Bx_validate = sigma*np.random.randn(len_validate, 1) + mu2x
By_validate = sigma*np.random.randn(len_validate, 1) + mu2y

XA_validate = np.append(Ax_validate,Ay_validate, axis=1)
XB_validate = np.append(Bx_validate,By_validate, axis=1)
X_validate = np.append(XA_validate,XB_validate, axis=0)

Y0_validate = np.zeros(shape=(len_validate, 1))
Y1_validate = np.ones(shape=(len_validate, 1))
Y_validate = np.append(Y0_validate,Y1_validate, axis=0)

####构图########################################
#定义训练路径
W = tf.Variable(initial_value=tf.zeros([2,1]), name='Weigh', dtype=tf.float32)
b = tf.Variable(initial_value=tf.zeros([1,1]), name='bias', dtype=tf.float32)
x = tf.placeholder(tf.float32, shape=(None,2), name='x')
y = tf.placeholder(tf.float32, shape=(None,1), name='y')

logist = tf.matmul(x, W) + b
pred = tf.sigmoid(logist)
loss = tf.reduce_mean(tf.square(y-pred))
optimizer = tf.train.GradientDescentOptimizer(step)
train = optimizer.minimize(loss=loss)
#定义验证路径
x_validate = tf.placeholder(tf.float32, shape=(None,2), name='x_validate')
y_validate = tf.placeholder(tf.float32, shape=(None,1), name='y_validate')
logist_validate = tf.matmul(x_validate, W) + b
pred_validate = tf.sigmoid(logist_validate, name = 'Pred')
isMatch = tf.equal(tf.cast(tf.greater(pred_validate, 0.5), tf.float32), y_validate)
accuracy = tf.reduce_mean(tf.cast(isMatch, tf.float32), name='Accuracy')

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) 
    for j in range(500):
        avloss = 0.
###################训练##############################
        total_batch = int(len_train*2/batch_train)
        for i in range(int(total_batch)):
            train_X = X[i*batch_train:(i+1)*batch_train, :]
            train_Y = Y[i*batch_train:(i+1)*batch_train, :]
            _, l = sess.run([train, loss], feed_dict={x: train_X, y: train_Y})
            avloss += l/total_batch
###################测试##############################
            if (j+1) % 100 == 0 and i == total_batch - 1:
                print('\n*************第', j+1, '次迭代*****************', '\n')
                print('损失: ', avloss, '\n')
                rw = sess.run(W)
                rb = sess.run(b)
                print('训练结果:\n')
                print('W=', rw, '\n\n', 'b=', rb, '\n')
                rAccuracy = sess.run(accuracy, feed_dict={x_validate:X_validate,y_validate:Y_validate})
                print('正确率 = ', rAccuracy)
                saver.save(sess, '.\model\my_test_model')

    

    

   


   

    
   

