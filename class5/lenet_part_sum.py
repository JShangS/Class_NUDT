# -*- coding:utf-8 -*-
# Author: Dengwen Lin
# Date: 2018/05/18

import numpy as np 
import tensorflow as tf
from tensorflow.contrib.layers import flatten 


    def _cnn_layer(self, scope_name, W_name, b_name, x, filter_shape, conv_strides, padding_tag='VALID'):
        with tf.name_scope(scope_name):
            conv_W = tf.get_variable(W_name, dtype=tf.float32, 
            	initializer=tf.truncated_normal(shape=filter_shape, mean=self.config.mu, stddev=self.config.sigma))

            conv_b = tf.get_variable(b_name, dtype=tf.float32, 
            	initializer=tf.zeros(filter_shape[3]))
            conv   = tf.nn.conv2d(x, conv_W, strides=conv_strides, padding=padding_tag) + conv_b
            
            with tf.name_scope('visual') as v_s:
                # scale weights to [0 1], type is still float
                x_min = tf.reduce_min(conv_W)
                x_max = tf.reduce_max(conv_W)
                kernel_0_to_1 = (conv_W - x_min) / (x_max - x_min)
                # to tf.image_summary format [batch_size, height, width, channels]
                kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])
                # this will display random 3 filters from the 64 in conv1
                show_img=tf.cond(tf.greater( filter_shape[2],1), lambda: tf.slice(kernel_transposed,[0,0,0,0],[-1,filter_shape[0],filter_shape[1],3]),lambda: kernel_transposed)
                tf.summary.image('conv_w', show_img, max_outputs=filter_shape[3])
                layer1_image1 = conv[0:1, :, :, 0:filter_shape[3]]
                layer1_image1 = tf.transpose(layer1_image1, perm=[3,1,2,0])
                tf.summary.image('feature', layer1_image1, max_outputs=16)
            
            return conv
    
 