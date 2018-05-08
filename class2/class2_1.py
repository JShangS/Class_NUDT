# 一. 变量定义、初始化:
#
# 1）定义三个变量：a,b,c。定义其初始化值initial_value均为0~9的随机正整数。
# 2）将a,b添加入key为'init'的collection中。并只初始化'init'这个集合中的变量。
# 3）此时fetch变量c, 观察出错报告并截图。找到出错原因及对应错误代号。
# 4）使用try-except 异常处理方法，收集未被初始化的变量，并将未初始化的变量再次初始化。
#
# 要求：不能用tf.global_variables_initializer()，按照步骤完成。提交代码和运行截图。

import tensorflow as tf
import numpy as np
import traceback
import string

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

a = tf.Variable(initial_value=np.random.randint(9), name='a')
tf.add_to_collection('init', a)
b = tf.Variable(initial_value=np.random.randint(9), name='b')
tf.add_to_collection('init', b)
c = tf.Variable(initial_value=np.random.randint(9), name='c')
d = tf.Variable(initial_value=np.random.randint(9), name='d')
f = c - d
get = tf.get_collection('init')
print(type(get))
success = False
atemp = 0
with tf.Session() as sess:
    while atemp<=3 and not success:
        try:
            sess.run(tf.variables_initializer(get))
            r = sess.run(get)
            rf = sess.run(f)
            # rc = sess.run(c)
            # rd = sess.run(d)
            success = True
        except tf.errors.FailedPreconditionError as error:
            atemp +=1
            # print('\n\n')
            # print('Error: ', error.message,'\n\n')
            indexstart = error.message.find('(')
            indexend = error.message.find(')')
            errorname = error.message[indexstart+1:indexend]
            errorname = errorname + ':0'
            print('未被初始化的变量：', errorname)
            get_v = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=errorname)
            # print('get_v:', get_v)
            sess.run(tf.variables_initializer(get_v))
            tf.add_to_collection('else_value', get_v)
            # get_else = tf.get_collection('else_value')
            # sess.run(tf.variables_initializer(list(flat(get_else))))
            print('初始化：', errorname, '->', sess.run(get_v))
    # r_else = sess.run(get_else)
    print(r, rf,'\n\n')


