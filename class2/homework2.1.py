import tensorflow as tf
import numpy as np

# 变量声明
a = tf.Variable(initial_value=np.random.randint(0, 9), name='a')
b = tf.Variable(initial_value=np.random.randint(0, 9), name='b')
c = tf.Variable(initial_value=np.random.randint(0, 9), name='c')
d = tf.Variable(initial_value=np.random.randint(0, 9), name='d')
h = tf.Variable(initial_value=np.random.randint(0, 9), name='h')
i = tf.Variable(initial_value=np.random.randint(0, 9), name='i')
f = tf.add(c, d)

tf.add_to_collection('init', a)
tf.add_to_collection('init', b)
init_var = tf.get_collection('init')
init = tf.variables_initializer(init_var)

sess = tf.Session()
sess.run(init)

# 将有可能出现未初始化变量的代码放入try中，在all_var集合中找到未初始化的变量，反复进行try except直到将所有未初始化的变量初始化
all_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# 一个表示是否可能出现未初始化异常的bool值
isUnInitExp = True
while isUnInitExp:
    isUnInitExp = False
    try:
        sess.run(f)
    except tf.errors.OpError as e:
        eMessage = e.message
        # 判断是否是未初始化的异常
        if eMessage[0:38] == "Attempting to use uninitialized value ":
            # 如果是未初始化引起的异常
            # 找到未初始化的变量的名字
            eMsgSplit = eMessage.split('\n')
            nameUnInitVar = eMsgSplit[0][38:] + ':0'
            # 将未初始化的变量的名字和all_var集合中的变量的名字比对，在all_var中找到未初始化的变量并且初始化
            for x in all_var:
                if nameUnInitVar == x.name:
                    initUnInit = tf.variables_initializer([x])
                    sess.run(initUnInit)
                    print("变量 name = " + nameUnInitVar + "已初始化")
                    break
            isUnInitExp = True
        else:
            # 如果不是未初始化引起的异常
            print("产生了不是未初始化变量引起的异常:")
            print(eMessage)
            isUnInitExp = False










