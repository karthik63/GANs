import tensorflow as tf
import numpy as np

def conv2d(a, b, c, d):
    return tf.nn.conv2d(a, b, c, d)

def concatenate_conditioning_vector_with_feature_map(x, y):
    x_shape = x.get_shape()
    y_shape = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shape[0], x_shape[1], x_shape[2], y_shape[3]])], axis=3)

def lrelu(x, leak=0.2):
    return tf.maximum(x*leak, x)


