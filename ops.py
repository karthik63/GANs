import tensorflow as tf
import numpy as np

def deconv2d(value, filter, output_shape, strides):
    return tf.nn.conv2d_transpose(value, filter, output_shape, strides)

def conv2d(a, b, c, d):
    return tf.nn.conv2d(a, b, c, d)

def concatenate_conditioning_vector_with_feature_map(x, y):
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shape[0], x_shape[1], x_shape[2], y_shape[3]])], axis=3)

def lrelu(x, leak=0.2):
    return tf.maximum(x*leak, x)

class batch_norm():
    def __init__(self, epsilon=1e-5, momentum=0.9, name='batch_norm'):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def bn(self, x, train=True):
        return tf.layers.batch_normalization(x,
                                             epsilon=self.epsilon,
                                             momentum=self.momentum,
                                             scale=True,
                                             training=train)