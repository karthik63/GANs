import tensorflow as tf
import numpy as np
import utils
import ops

class DCGAN:
    def __init__(self, sess, n_epochs, learning_rate, beta1, train_size, batch_size, input_height, input_width, output_height,
                 output_width, dataset, input_fname_pattern, chckpoint_dir, sample_dir, train, test, crop, visualise,
                 y_dim=1, g_filter_dim=64, g_fc_dim=1024, d_filter_dim=64, d_fc_dim=1024, c_dim=1):
        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.c_dim = c_dim
        self.y_dim = y_dim
        self.g_filter_dim = g_filter_dim
        self.g_fc_dim = g_fc_dim
        self.d_filter_dim = d_filter_dim
        self.d_fc_dim = d_fc_dim

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.train_size = train_size

        self.dataset = dataset
        self.input_fname_pattern = input_fname_pattern
        self.chckpoint_dir = chckpoint_dir
        self.sample_dir = sample_dir

        self.train = train
        self.test = test
        self.visualise = visualise
        #haven't done batch normalisation

        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder('float', [self.batch_size, self.input_height, self.input_width, self.y_dim],
                                     name='input_placeholder')

    def discriminator(self, image, Y, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

        Yb = tf.reshape(Y, [self.batch_size, 1, 1, self.c_dim])
        l0 = ops.concatenate_conditioning_vector_with_feature_map(image, Yb)

        w1 = tf.get_variable(name='filter_1',
                             shape=[self.batch_size, 5, 5, self.c_dim + self.y_dim, self.c_dim + self.y_dim],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        b1 = tf.get_variable(name='biases_filter1',
                             shape=[self.c_dim + self.y_dim, self.c_dim + self.y_dim],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        l1 = ops.conv2d(l0, w1, [1, 2, 2, 1], 'SAME')
        l1 = tf.add(l1, b1)
        l1 = tf.nn.relu(l1)
        l1 = ops.concatenate_conditioning_vector_with_feature_map(l1, Yb)

        w2 = tf.get_variable(name='filter_2',
                             shape=[self.batch_size, 5, 5, self.c_dim + 2*self.y_dim, self.d_filter_dim],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        b2 = tf.get_variable(name='biases_filter2',
                             shape=[self.c_dim + 2*self.y_dim, self.d_filter_dim],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        l2 = ops.conv2d(l1, w2, [1, 2, 2, 1], 'SAME')
        l2 = tf.add(l2, b2)
        l2 = tf.nn.relu(l2)
        l2 = tf.reshape(l2, [self.batch_size, self.d_filter_dim * self.input_width * self.input_height])
        l2 = tf.concat([l2, Y], axis=1)

        w3 = tf.get_variable(name='fc_1',
                             shape=[self.d_filter_dim * self.input_width * self.input_height + self.y_dim, self.d_fc_dim],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        b3 = tf.get_variable(name='biases_fc1',
                             shape=[self.d_fc_dim],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        l3 = tf.matmul(l2, w3)
        l3 = tf.add(l3, b3)
        l3 = tf.nn.relu(l3)
        l3 = tf.concat([l3, Y], axis=1)

        w4 = tf.get_variable(name='fc2',
                             shape=[self.d_fc_dim, 1],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        b4 = tf.get_variable(name='biases_fc2',
                             shape=[1, 1],
                             dtype='float',
                             initializer=tf.truncated_normal_initializer())
        l4 = tf.matmul(l3, w4)
        l4 = tf.add(l4, b4)

        return l4, tf.nn.sigmoid(l4)
