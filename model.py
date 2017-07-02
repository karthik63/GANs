import tensorflow as tf
import numpy as np
import utils
from argument_parser import ArgumentParser
from config import Config
import ops
from math import ceil
from scipy.misc import toimage

np.random.seed(4321)
tf.set_random_seed(1234)
class DCGAN:
    def __init__(self, config, sess, y_dim=10, z_dim=100, g_filter_dim=64, g_fc_dim=1024,
                 d_filter_dim=64, d_fc_dim=1024, c_dim=1, input_size=50000):
        self.sess = sess
        self.config = config
        self.crop = config.crop

        self.batch_size = config.batch_size

        self.input_size = input_size

        self.input_height = config.input_height
        self.input_width = config.input_width
        self.output_height = config.output_height
        self.output_width = config.output_width

        self.c_dim = c_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.g_filter_dim = g_filter_dim
        self.g_fc_dim = g_fc_dim
        self.d_filter_dim = d_filter_dim
        self.d_fc_dim = d_fc_dim

        self.n_epochs = config.n_epochs
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.train_size = config.train_size

        self.dataset = config.dataset
        self.input_fname_pattern = config.input_fname_pattern
        self.checkpoint_dir = config.checkpoint_dir
        self.sample_dir = config.sample_dir

        self.train = config.train
        self.visualise = config.visualise
        #haven't done batch normalisation

        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')

        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')

        self.resize = config.resize

        self.build_model()

    def build_model(self):
        self.y = tf.placeholder('float', [None, self.y_dim], name='y')
        self.z = tf.placeholder('float', [None, self.z_dim], name='z')
        self.input_images = tf.placeholder('float', [self.batch_size, self.input_height, self.input_width, self.c_dim])

        self.D = self.discriminator(self.input_images, self.y)
        self.G = self.generator(self.z, self.y)
        self.D_ = self.discriminator(self.G, self.y, True)

        self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=tf.ones_like(self.D)))
        self.d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_real_loss + self.d_fake_loss
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_, labels=tf.ones_like(self.D_)))

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.saver = tf.train.Saver()

    def train_model(self):

        if self.dataset == 'mnist':
            mnist_utils = utils.MNISTUtils('mnist.pkl.gz', self.config)
        elif self.dataset == 'fashion':
            mnist_utils = utils.FashionUtils('./fashion_clothing_and_category', self.config)

        self.d_optimiser = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optimiser = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(self.g_loss, var_list=self.g_vars)

        writer = tf.summary.FileWriter('./graphs/lel')
        writer.add_graph(self.sess.graph)

        image_index = 0
        input_size = self.input_size

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.n_epochs):
            epoch_loss_d = 0
            epoch_loss_g = 0
            for i in range(int(ceil(input_size/self.batch_size))):
                image_index = i * self.batch_size
                current_batch_size = min(input_size - image_index, self.batch_size)
                current_batch_X = np.array([ mnist_utils.get_image(image_index) for image_index in range(image_index, image_index + current_batch_size)])
                current_batch_Y = np.array([ mnist_utils.get_label(image_index) for image_index in range(image_index, image_index + current_batch_size)])
                current_batch_Z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim])

                temp1, temp2 = self.sess.run([self.d_loss, self.g_loss], feed_dict={self.input_images: current_batch_X, self.y: current_batch_Y, self.z: current_batch_Z})
                _ = self.sess.run(self.d_optimiser, feed_dict={self.input_images: current_batch_X, self.y: current_batch_Y, self.z: current_batch_Z})
                _ = self.sess.run(self.g_optimiser, feed_dict={self.input_images: current_batch_X, self.y: current_batch_Y, self.z: current_batch_Z})
                _ = self.sess.run(self.g_optimiser, feed_dict={self.input_images: current_batch_X, self.y: current_batch_Y, self.z: current_batch_Z})
                gen_image = self.sess.run(self.G, feed_dict={self.y: current_batch_Y, self.z: current_batch_Z})

                toimage(gen_image[3]).save('./generated_images/' + 'e' + str(epoch) + 'b' + str(i) + '.png')
                epoch_loss_d += temp1
                epoch_loss_g += temp2

                print('epoch: ' + str(epoch) + ' batch: ' + str(i) + ' disc loss :' + str(temp1/current_batch_size) + ' gen loss :' + str(temp2/current_batch_size))

            print('epoch no : ' + str(epoch) + ' discriminator loss is : ' + str(epoch_loss_d/self.input_size) +
                  ' generator loss is : ' + str(epoch_loss_g/self.input_size) + ' no ' +
                  str(mnist_utils.get_label(image_index + 2)))

            toimage(gen_image[2]).save('./epoch_images/' + 'epoch' + str(epoch) + '.png')

        self.saver.save(self.sess, './weights1/single_epoch')

    def discriminator(self, image, Y, reuse=False):

        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            Yb = tf.reshape(Y, [self.batch_size, 1, 1, self.y_dim])
            l0 = ops.concatenate_conditioning_vector_with_feature_map(image, Yb)

            w1 = tf.get_variable(name='d_filter_1',
                                 shape=[5, 5, self.c_dim + self.y_dim, self.c_dim + self.y_dim],
                                 dtype='float',
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable(name='d_biases_filter1',
                                 shape=[1, self.c_dim + self.y_dim],
                                 dtype='float',
                                 initializer=tf.constant_initializer(0.0))
            l1 = ops.conv2d(l0, w1, [1, 2, 2, 1], 'SAME')
            l1 = tf.add(l1, b1)
            l1 = tf.nn.relu(l1)
            l1 = ops.concatenate_conditioning_vector_with_feature_map(l1, Yb)

            w2 = tf.get_variable(name='d_filter_2',
                                 shape=[5, 5, self.c_dim + 2*self.y_dim, self.d_filter_dim],
                                 dtype='float',
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable(name='d_biases_filter2',
                                 shape=[1, self.d_filter_dim],
                                 dtype='float',
                                 initializer=tf.constant_initializer(0.0))
            l2 = ops.conv2d(l1, w2, [1, 2, 2, 1], 'SAME')
            l2 = self.d_bn1.bn(tf.add(l2, b2))
            l2 = tf.nn.relu(l2)
            l2 = tf.reshape(l2, [self.batch_size, -1])
            l2 = tf.concat([l2, Y], axis=1)

            w3 = tf.get_variable(name='d_fc_1',
                                 shape=[l2.get_shape()[1], self.d_fc_dim],
                                 dtype='float',
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable(name='d_biases_fc1',
                                 shape=[1, self.d_fc_dim],
                                 dtype='float',
                                 initializer=tf.constant_initializer(0.0))
            l3 = tf.matmul(l2, w3)
            l3 = self.d_bn2.bn(tf.add(l3, b3))
            l3 = tf.nn.relu(l3)
            l3 = tf.concat([l3, Y], axis=1)

            w4 = tf.get_variable(name='d_fc2',
                                 shape=[self.d_fc_dim + self.y_dim, 1],
                                 dtype='float',
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable(name='d_biases_fc2',
                                 shape=[1, 1],
                                 dtype='float',
                                 initializer=tf.constant_initializer(0.0))
            l4 = tf.matmul(l3, w4)
            l4 = tf.add(l4, b4)

        return l4

    def generator(self, z, Y, reuse=False):

        with tf.variable_scope('generator'):

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = int(ceil(float(s_h) / 2)), int(ceil(float(s_w) / 2))
            s_h4, s_w4 = int(ceil(float(s_h2) / 2)), int(ceil(float(s_w2) / 2))

            Yb = tf.reshape(Y, [self.batch_size, 1, 1, self.y_dim])
            l0 = tf.concat([z, Y], axis=1)

            w1 = tf.get_variable(name='g_weights_fc1',
                                 dtype='float',
                                 shape=[self.z_dim + self.y_dim, self.g_fc_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b1 = tf.get_variable(name='g_biases_fc1',
                                 dtype='float',
                                 shape=[1, self.g_fc_dim],
                                 initializer=tf.constant_initializer(0.0))
            l1 = tf.matmul(l0, w1)
            l1 = self.g_bn0.bn(tf.add(l1, b1))
            l1 = tf.nn.relu(l1)
            l1 = tf.concat([l1, Y], axis=1)

            w2 = tf.get_variable(name='g_weights_fc2',
                                 dtype='float',
                                 shape=[self.g_fc_dim + self.y_dim, self.g_filter_dim*2*s_h4*s_w4],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b2 = tf.get_variable(name='g_biases_fc2',
                                 dtype='float',
                                 shape=[1, self.g_filter_dim*2*s_h4*s_w4],
                                 initializer=tf.constant_initializer(0.0))
            l2 = tf.matmul(l1, w2)
            l2 = self.g_bn1.bn(tf.add(l2, b2))
            l2 = tf.nn.relu(l2)
            l2 = tf.reshape(l2, [self.batch_size, s_h4, s_w4, self.g_filter_dim*2])
            l2 = ops.concatenate_conditioning_vector_with_feature_map(l2, Yb)

            w3 = tf.get_variable(name='g_weights_filter1',
                                 dtype='float',
                                 shape=[5, 5, self.g_filter_dim*2, self.g_filter_dim*2+self.y_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b3 = tf.get_variable(name='g_biases_filter1',
                                 dtype='float',
                                 shape=[1, self.g_filter_dim*2],
                                 initializer=tf.constant_initializer(0.0))
            l3 = ops.deconv2d(l2, w3, [self.batch_size, s_h2, s_w2, self.g_filter_dim * 2], [1, 2, 2, 1])
            l3 = self.g_bn2.bn(tf.add(l3, b3))
            l3 = tf.nn.relu(l3)
            l3 = ops.concatenate_conditioning_vector_with_feature_map(l3, Yb)

            w4 = tf.get_variable(name='g_weights_filter2',
                                 dtype='float',
                                 shape=[5, 5, self.c_dim, self.g_filter_dim*2 + self.y_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
            b4 = tf.get_variable(name='g_biases_filter2',
                                 dtype='float',
                                 shape=[1, self.c_dim],
                                 initializer=tf.constant_initializer(0.0))
            l4 = ops.deconv2d(l3, w4, [self.batch_size, s_h, s_w, self.c_dim], [1, 2, 2, 1])
            l4 = tf.add(l4, b4)

            return tf.nn.sigmoid(l4)

    def generate_image(self, z, Y):
        self.saver.restore(self.sess, './weights1/single_epoch')
        temp_y = np.zeros([50]).astype('int')
        temp_y[6] = 1
        generated_image = self.sess.run(self.G, feed_dict={self.z: np.random.uniform(-1, 1, [1, self.z_dim]),
                                                           self.y: temp_y})

        toimage(np.reshape(generated_image, [28, 28])).save('./predicted_images/' + 'img1' + '.png')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
config.gpu_options.allow_growth = True
config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=config) as sess:

    args = ArgumentParser().args

    config = Config(args)

    gen1 = DCGAN(config, sess, y_dim=50, z_dim=100, g_filter_dim=64, g_fc_dim=1024,
                 d_filter_dim=64, d_fc_dim=1024, c_dim=1, input_size=50000)

    #gen1.discriminator(None, None, False)
    gen1.train_model()

    #gen1.generate_image(np.random.uniform(-1, 1, [100, 100]), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]))
