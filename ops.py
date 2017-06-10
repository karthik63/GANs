import tensorflow as tf
import numpy as np

def conv2d(a, b, c, d):
    return tf.nn.conv2d(a, b, c, d)

def concatenate_conditioning_vector_with_feature_map(x, y):
    return tf.concat([x, y*tf.constant([1])], axis=3)

def lrelu(x, leak=0.2):
    return tf.maximum(x*leak, x)


