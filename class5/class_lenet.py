import tensorflow as tf
import numpy as np


class Lennet(object):
    #############构造函数#######################
    def __init__(self, NumClass):
        with tf.variable_scope("lenet_var") as scope:
            self.raw_input_image = tf.placeholder(tf.float32, [None, 1024])
            self.input_x = tf.reshape(
            self.raw_input_image, [-1, 32, 32, 1], name='input')
            self.input_y = tf.placeholder(tf.float32, [None, 10],name='label')
            self.NumClass = NumClass
            self.logist = self.buildCNN(self.input_x, self.NumClass)
            scope.reuse_variables()
            tf.summary.image('inputx', self.input_x)
            tf.summary.histogram('labely', tf.argmax(self.input_y, 1))

    ####################全连接#####################################
    def fcLayer(self,
                name,
                input,
                inputSize,
                outputSize,
                mu=0,
                sigma=0.1,
                activeFunction=None):
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                initializer=tf.truncated_normal(
                    shape=([inputSize, outputSize]), mean=mu, stddev=sigma),
                name='w')
            b = tf.get_variable(
                initializer=tf.truncated_normal(
                    shape=([outputSize]), mean=mu, stddev=sigma),
                name='b')
            xwb = tf.nn.xw_plus_b(input, w, b, name=scope.name)
            # output = tf.nn.relu(xwb)
            if activeFunction is None:
                output = xwb
            else:
                output = activeFunction(xwb)
            ###############tensorboard####################
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", output)
            #####################################
        return output

    ####################卷积层#####################################
    def convLayer(self,
                  name,
                  input,
                  kHeight,
                  kWeight,
                  featureNum,
                  mu=0.0,
                  sigma=0.1,
                  stridX=int(1),
                  stridY=int(1),
                  padding='SAME',
                  activeFunction=None):  ####data_format='NHWC'默认的
        channel = int(input.get_shape()[-1])
        # channel = int(1)
        # print('channel:', channel)
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                initializer=tf.truncated_normal(
                    shape=([kHeight, kWeight, channel, featureNum]),
                    mean=mu,
                    stddev=sigma),
                name='w')
            b = tf.get_variable(
                initializer=tf.truncated_normal(
                    shape=([featureNum]), mean=mu, stddev=sigma),
                name='b')
            featureMap = tf.nn.conv2d(
                input, w, strides=[1, stridX, stridY, 1], padding=padding)
            xwb = tf.nn.bias_add(featureMap, b, name=scope.name)
            # output = tf.nn.relu(xwb)
            if activeFunction is None:
                output = xwb
            else:
                output = activeFunction(xwb)
            ###############tensorboard####################
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activation", output)
            #####################################
            return output

    #################池化层#############################
    def maxPooling(self,
                   name,
                   input,
                   kHeight,
                   kWeight,
                   stridX=2,
                   stridY=2,
                   padding='SAME'):
        return tf.nn.max_pool(
            input,
            ksize=[1, kHeight, kWeight, 1],
            strides=[1, stridX, stridY, 1],
            padding=padding)

    ##############画图########################
    def buildCNN(self, input, NumClass, isTrained=True):
        with tf.name_scope('lenet') as scope:
            mu = 0.0
            sigma = 0.1
            conv1 = self.convLayer(
                'conv1',
                input,
                5,
                5,
                6,
                mu=mu,
                sigma=sigma,
                padding='VALID',
                activeFunction=tf.nn.relu)
            #Layer1, pooling：input:28*28*6,output: 14x14x6
            pool1 = self.maxPooling('pool1', conv1, 2, 2, padding='VALID')
            #Layer2, convolution: input:14x14x6,output: 10x10x16
            conv2 = self.convLayer(
                'conv2',
                pool1,
                5,
                5,
                16,
                mu=mu,
                sigma=sigma,
                padding='VALID',
                activeFunction=tf.nn.relu)
            #Layer2, pooling：input:10x10x16,output: 5x5x16.
            pool2 = self.maxPooling('pool2', conv2, 2, 2, padding='VALID')
            #vectoring, Input = 5x5x16. Output = 400.
            fc0 = tf.reshape(pool2, shape=[-1, 400])
            #Layer3, Fully Connected. Input = 400. Output = 120.
            fc1 = self.fcLayer(
                'fc1',
                fc0,
                400,
                120,
                mu=mu,
                sigma=sigma,
                activeFunction=tf.nn.relu)
            #Layer4, Fully Connected. Input = 120. Output = 84.
            fc2 = self.fcLayer('fc2', fc1, 120, 84, mu=mu, sigma=sigma)
            #Layer5, Fully Connected. Input = 84. Output = 10.
            logist = self.fcLayer(
                'logist', fc2, 84, NumClass, mu=mu, sigma=sigma)
            tf.summary.histogram('logist',logist)
        return logist