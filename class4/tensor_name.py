import tensorflow as tf
import numpy as np

with tf.name_scope('n_1'):
    with tf.name_scope('n_2'):
        with tf.variable_scope('v_1'):
            with tf.variable_scope('v_2'):
                Weights1 = tf.get_variable('Weights', shape=[2,3])
                bias1 = tf.Variable([0.52], name='bias')


print (Weights1.name)
print (bias1.name)

with tf.name_scope('n_1'):
    with tf.name_scope('n_2'):
        with tf.variable_scope('v_1'):
            with tf.variable_scope('v_2'):
                bias1 = tf.Variable([0.52], name='bias')

print (bias1.name)

with tf.variable_scope('v_scope') as scope1:
    Weights1 = tf.get_variable('Weights', shape=[2,3])
    bias1 = tf.Variable([0.52], name='bias')

# 下面来共享上面已经定义好的变量
# note: 在下面的 scope 中的get_variable()变量必须已经定义过了，才能设置 reuse=True，否则会报错
with tf.variable_scope('v_scope', reuse=True) as scope2:
    Weights2 = tf.get_variable('Weights')
    bias2 = tf.Variable([0.52], name='bias')
    bias3 = tf.Variable([0.52], name='bias')

print (Weights1.name)
print (Weights2.name)
print (bias1.name)
print (bias2.name)
print (bias3.name)

with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

print (weights1.name)
print (weights2.name)

with tf.name_scope('conv1') as scope:
    weights1 = tf.Variable([1.0, 2.0], name='weights')
    bias1 = tf.Variable([0.3], name='bias')

with tf.name_scope('conv2') as scope:
    weights2 = tf.Variable([4.0, 2.0], name='weights')
    bias2 = tf.Variable([0.33], name='bias')

print (weights1.name)
print (weights2.name)



np.random.seed(1)
b=tf.Variable(initial_value=np.random.randint(10),name='b')
a=tf.Variable(initial_value=np.random.randint(10),name='a')
c=tf.Variable(initial_value=np.random.randint(10),name='b')


add0=tf.add(a,b)
print(add0)
add1=tf.add(a,b, name="add")
print(add1)
add2=tf.add(a,c, name="add")
print(add2)

t=tf.get_default_graph().get_tensor_by_name("add:0")
print(t)

# ��value�� is a tensor with shape [5, 30]
# Split ��value�� into 3 tensors along dimension 1

v_split=tf.Variable(initial_value=np.random.rand(5,30),name='v_split')
all_v =tf.global_variables()  


split0, split1, split2 = tf.split( v_split, 3,1, name='split_n')
tf.shape(split0) 
print(split0)
print(split1)
print(split2)
#==> [5, 10]

s_input=tf.constant([[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]],name='s_input')
#��input�� is 
#[[[1, 1, 1], [2, 2, 2]],[[3, 3, 3], [4, 4, 4]],[[5, 5, 5], [6, 6, 6]]]
s1=tf.slice(s_input, [1, 0, 0], [1, 1, 3]) #==> [[[3, 3, 3]]]
print(s1)
s2=tf.slice(s_input, [1, 0, 0], [1, 2, 3]) 
print(s2)
#==> 
#[[[3, 3, 3],
#[4, 4, 4]]]
s3=tf.slice(s_input, [1, 0, 0], [2, 1, 3]) #==> 
print(s3)
#[[[3, 3, 3]],
#[[5, 5, 5]]]

for i in all_v:  
    print  (i)

tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
for tensor_name in tensor_name_list:
    print(tensor_name,'\n')

