import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()
mnist = input_data.read_data_sets('./data/mnist',one_hot=False)
print(mnist.validation.num_examples)
print(mnist.train.num_examples)
print(mnist.test.num_examples)

def minist_draw(im):
    im = im.reshape(28, 28)
    fig = plt.figure()
    plotwindow = fig.add_subplot(111)
    plt.axis('off')
    plt.imshow(im, cmap='gray')
    plt.show()
    # plt.savefig("test.png")  # ������ļ�
    plt.close()

def balanced_batch(batch_x,batch_y, num_cls):
    batch_size=len(batch_y)
    pos_per_cls_e=round(batch_size/2/num_cls/2)
    pos_per_cls_e=2
    index=batch_y.argsort()
    ys_1=batch_y[index]
    #print(ys_1)
    
    num_class=[]
    pos_samples=[]
    neg_samples=set()
    cur_ind=0
    for item in set(ys_1):
        num_class.append((ys_1==item).sum())
        num_pos=pos_per_cls_e
        while(num_pos>num_class[-1]):
            num_pos-=2
        pos_samples.extend(np.random.choice(index[cur_ind:cur_ind+num_class[-1]],num_pos,replace=False).tolist())
        neg_samples=neg_samples|(set(index[cur_ind:cur_ind+num_class[-1]])-set(list(pos_samples)))
        cur_ind+=num_class[-1]
    
    neg_samples=list(neg_samples)
    
    x1_index=pos_samples[::2]
    x2_index=pos_samples[1:len(pos_samples)+1:2]

    x1_index.extend(neg_samples[::2])
    x2_index.extend(neg_samples[1:len(neg_samples)+1:2])
    
    p_index=np.random.permutation(len(x1_index))
    x1_index=np.array(x1_index)[p_index]
    x2_index=np.array(x2_index)[p_index]

    r_x1_batch=batch_x[x1_index]
    r_x2_batch=batch_x[x2_index]
    r_y_batch=np.array(batch_y[x1_index]!=batch_y[x2_index],dtype=np.float32)
    return r_x1_batch,r_x2_batch,r_y_batch

lr = 0.01
iterations = 20000
batch_size = 64

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for itera in range(iterations):

        batch_x, batch_y = mnist.train.next_batch(batch_size*2)

        xs_1,xs_2,y_s=balanced_batch(batch_x,batch_y, num_cls=10)

        for i in range(10):
            minist_draw(xs_1[i])
            minist_draw(xs_2[i])
