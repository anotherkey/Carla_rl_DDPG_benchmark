from __future__ import print_function

import os

import scipy
from scipy.misc import imresize

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

from .agent import Agent


class DataProcess(Agent):
    def __init__(self, sess, image_cut=(115, 510)):

        Agent.__init__(self)
        # self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
        self.dropout_vec = [1.0] * 8 + [0.7] * 2

        self._image_size = (88, 200, 3)
        self._image_cut = image_cut

        self._sess = sess

        self._input_images = tf.placeholder("float",
                                            shape=[None, self._image_size[0],
                                                         self._image_size[1],
                                                         self._image_size[2]],
                                            name="input_image")

        self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

        with tf.name_scope("Network"):
        # with tf.variable_scope("Network"):
            self._network_tensor = load_network(self._input_images,
                                                self._image_size,
                                                self._dout)

        dir_path = os.path.dirname(__file__)
        self._models_path = dir_path + '/model/'
        self.variables_to_restore = tf.global_variables()

    def load_model(self):

        saver = tf.train.Saver(self.variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self._sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt

    def compute_feature(self, sensor_data):

        rgb_image = sensor_data['CameraRGB'].data
        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0], self._image_size[1]])

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        feedDict = {self._input_images: image_input, self._dout: [1] * len(self.dropout_vec)}

        # tf.reset_default_graph()
        output_vector = self._sess.run(self._network_tensor, feed_dict=feedDict)
        return output_vector[0]


# 权重初始化
def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    # Variable代表一个可修改的张量

    # tf.Variable()创建变量时，name属性值允许重复，检查到相同名字的变量时，由自动别名机制创建不同的变量。
    # 返回一个由initial创建的变量
    return tf.Variable(initial)


# Xavier初始化器的作用，就是在初始化深度学习网络得时候让权重不大不小。
# http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf
def weight_xavi_init(shape, name):
    # get_variable如果已存在参数定义相同的变量，就返回已存在的变量，否则创建由参数定义的新变量。
    # with tf.variable_scope('W_conv', reuse=True):

    initial = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    # 该函数返回一个用于初始化权重的初始化程序 “Xavier” 。
    # 这个初始化器是用来保持每一层的梯度大小都差不多相同。
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):

    def __init__(self, dropout, image_shape):
        """ We put a few counters to see how many times we called each function """
        # 计数
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1
        # get_shape数据类型只能是tensor,且返回的是一个元组（tuple）
        # 卷积层的输入 filters_in == 3
        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, name='W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)
        # 通过placeholder输入的项数，步长，步长，颜色通道数
        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    # Batch Normalization通过减少内部协变量加速神经网络的训练。
    # 可以用作conv2d和fully_connected的标准化函数。
    def bn(self, x):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]

        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x)
            x = self.dropout(x)

            return self.activation(x)

    def fc_block(self, x, output_size):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features


def load_network(input_image, input_size, dropout):
    branches = []

    x = input_image

    network_manager = Network(dropout, tf.shape(x))

    """conv1"""  # kernel sz, stride, num feature maps
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID')
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID')
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID')
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID')
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    #  np.prod()函数用来计算所有元素的乘积
    x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x)

    """ fc1 """
    x = network_manager.fc_block(x, 512)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512)

    return x


