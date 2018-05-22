import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
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
def calc_accuracy(pred, threshold, y_test):
    accuracy = tf.equal(tf.to_float(pred > threshold), y_test)
    # accuracy = tf.equal(pred, y)
    accuracy = tf.cast(accuracy, tf.float32)
    accuracy = tf.reduce_mean(accuracy)
    return restore_sess.run(
        accuracy, feed_dict={
            x1_test: xs_t1,
            x2_test: xs_t2,
            y_test: y_ts
        })


##########读取数据################################
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
mnist = input_data.read_data_sets('./data/mnist', one_hot=False)
#######################恢复图##############################
batch_size = 64
tf.reset_default_graph()
restore_graph = tf.Graph()
with tf.Session(graph=restore_graph) as restore_sess:
    restore_saver = tf.train.import_meta_graph(
        './model_class3/my_test_model.meta')
    restore_saver.restore(restore_sess,
                          tf.train.latest_checkpoint('./model_class3'))
    graph = tf.get_default_graph()
    x1_test = graph.get_tensor_by_name('input1:0')
    x2_test = graph.get_tensor_by_name('input2:0')
    y_test = graph.get_tensor_by_name('label:0')
    # W1 = graph.get_tensor_by_name('W1:0')
    # b1 = graph.get_tensor_by_name('B1:0')
    # W2 = graph.get_tensor_by_name('W2:0')
    # b2 = graph.get_tensor_by_name('B2:0')
    pred = graph.get_tensor_by_name('Ew:0')
    accuracy = graph.get_tensor_by_name('Accuracy:0')
    #######################测试#############################
    ACC = []
    for i in range(100):
        batch_x, batch_y = mnist.test.next_batch(batch_size * 2)
        xs_t1, xs_t2, y_ts = balanced_batch(batch_x, batch_y, num_cls=10)
        ACC = np.append(ACC, calc_accuracy(pred, 2.4, y_test))
        # ACC = restore_sess.run(
        #     accuracy, feed_dict={
        #         x1_test: xs_t1,
        #         x2_test: xs_t2,
        #         y_test: y_ts
        #     })
    print('准确率 ACC =', restore_sess.run(tf.reduce_mean(ACC)))
    ##############门限和检测概率曲线#######################
    # threshold_range = np.linspace(0, 10, 101)
    # ACC_mean = []
    # for it in threshold_range:
    #     print(it)
    #     ACC = []
    #     for i in range(100):
    #         batch_x, batch_y = mnist.test.next_batch(batch_size * 2)
    #         xs_t1, xs_t2, y_ts = balanced_batch(batch_x, batch_y, num_cls=10)
    #         ACC = np.append(ACC, calc_accuracy(pred, it, y_test))
    #     ACC_mean = np.append(ACC_mean, restore_sess.run(tf.reduce_mean(ACC)))
    # np.save('classe_test_ACC.npy', ACC_mean)
    # np.save('classe_test_threshold.npy', threshold_range)
    # plt.plot(threshold_range, ACC_mean)
    # plt.xlabel("Threshold")
    # plt.ylabel("Accuracy")
    # plt.grid(True)
    # plt.show()
    ##############ROC曲线#######################
    
tf.logging.set_verbosity(old_v)
