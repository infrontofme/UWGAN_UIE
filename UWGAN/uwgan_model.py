"""
Some codes from https://github.com/kskin/WaterGAN
"""
import os
from glob import glob
import time
import scipy
from scipy import misc
import numpy as np
import scipy.io as sio
from six.moves import xrange

from ops import *


class UWGAN(object):

    def __init__(self, sess, input_height=640, input_width=480, output_height=256, output_width=256,
                 is_crop=True, batch_size=64, y_dim=None, z_dim=100, df_dim=64, dfc_dim=1024, c_dim=3, save_epoch=10,
                 water_dataset_name='default', air_dataset_name='default', depth_dataset_name='default',
                 input_fname_pattern=('*.png', '*.jpg'), checkpoint_dir=None, results_dir=None):
        """
        Args:
            :param sess: Tensorflow session
            :param input_height: input image data height
            :param input_width: input image data width
            :param output_height: output image data height
            :param output_width: output image data width
            :param is_crop: True for training, False for testing
            :param batch_size: The size of batch. Should be specified before training
            :param y_dim: (optional) Dimension of dim for y. [None]
            :param z_dim: (optional) Dimension of dim for z. [100]
            :param df_dim: (optional) Dimension of Discriminator filters in first conv layers. [64]
            :param dfc_dim: (optional) Dimension of Discriminator units for fully connected layer. [1024]
            :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
            :param save_epoch: The epoch to save the model
            :param water_dataset_name: The name of water dataset
            :param air_dataset_name: The name of air dataset
            :param depth_dataset_name: The name of depth dataset, corresponding to air dataset
            :param input_fname_pattern: (*.jpg, *.png) glob pattern of filename of input images
            :param checkpoint_dir: Directory name to save the checkpoints [checkpoint]
            :param results_dir: Directory name to save the checkpoints [results]
        """

        self.sess = sess
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.save_epoch = save_epoch

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.df_dim = df_dim
        self.dfc_dim = dfc_dim

        self.water_dataset_name = water_dataset_name
        self.air_dataset_name = air_dataset_name
        self.depth_dataset_name = depth_dataset_name
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir

        self.sw = 256
        self.sh = 256

        # batch normalization: deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')
            self.g_bn4 = batch_norm(name='g_bn4')

        # build model
        if self.y_dim:
            y = tf.placeholder(tf.float32, shape=[self.batch_size, self.y_dim], name='y')
            print(y)
        self.image_dims = [self.output_height, self.output_width, self.c_dim]
        self.D_image_dims = [self.output_height, self.output_width, self.c_dim]
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)
        self.water_inputs = tf.placeholder(tf.float32, shape=[self.batch_size] + self.D_image_dims, name='real_images')
        self.air_inputs = tf.placeholder(tf.float32, shape=[self.batch_size] + self.image_dims, name='air_images')
        self.depth_inputs = tf.placeholder(tf.float32,
                                           shape=[self.batch_size, self.output_height, self.output_width, 1],
                                           name='depth')

        # JM model Generator
        self.G, self.eta_r, self.eta_g, self.eta_b, self.A, self.B = self.wc_generator(self.z, self.air_inputs, self.depth_inputs)

        self.D, self.D_logits = self.discriminator(image=self.water_inputs)

        self.wc_sampler = self.wc_sample(z=self.z, image=self.air_inputs, depth=self.depth_inputs)
        # self.D_G = tf.concat([self.G, self.eta_d_bs], axis=3)
        print("G Input to D: ", end=' ')
        print(self.G)
        self.D_, self.D_logits_ = self.discriminator(image=self.G, reuse=True)
        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G, max_outputs=200)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits, labels=tf.ones_like(self.D)))

        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.zeros_like(self.D_)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.eta_r_loss = -tf.minimum(tf.reduce_min(self.eta_r), 0) * 10000
        self.eta_g_loss = -tf.minimum(tf.reduce_min(self.eta_g), 0) * 10000
        self.eta_b_loss = -tf.minimum(tf.reduce_min(self.eta_b), 0) * 10000
        self.A_loss = -tf.minimum(tf.reduce_min(self.A), 0) * 10000
        self.B_loss = -tf.minimum(tf.reduce_min(self.B), 0) * 10000

        self.g_loss = self.g_loss + self.eta_r_loss + self.eta_g_loss + self.eta_b_loss + self.A_loss + self.B_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        tf.summary.scalar("D_realdata", self.D)
        tf.summary.scalar("D_fakedata", self.D_)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()

    def train(self, config):
        """
        Training W-gan
        :param config: parameter config
        :return:
        """
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.global_variables_initializer().run()

        g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        writer = tf.summary.FileWriter(config.log_dir, self.sess.graph)

        # Starting training
        counter = 1
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load Failed...")

        water_data = sorted(glob(os.path.join(
            "./data", config.water_dataset, '*.jpg')))
        print(len(water_data))
        air_data = sorted(glob(os.path.join(
            "./data", config.air_dataset, self.input_fname_pattern[0])))
        print(len(air_data))
        depth_data = sorted(glob(os.path.join(
            "./data", config.depth_dataset, "*.mat")))
        print(len(depth_data))

        for epoch in xrange(config.epoch+1):
            water_batch_idxs = min(min(len(air_data), len(water_data)), config.train_size) // config.batch_size
            randombatch = np.arange(water_batch_idxs * config.batch_size)
            np.random.shuffle(randombatch)

            #Load water images
            for idx in xrange(0, (water_batch_idxs*config.batch_size), config.batch_size):
                water_batch_files = []
                air_batch_files = []
                depth_batch_files = []

                for id in xrange(0, config.batch_size):
                    water_batch_files = np.append(water_batch_files, water_data[randombatch[idx + id]])
                    air_batch_files = np.append(air_batch_files, air_data[randombatch[idx + id]])
                    depth_batch_files = np.append(depth_batch_files, depth_data[randombatch[idx + id]])
                # print(depth_batch_files)

                air_batch = [self.read_img(air_batch_file) for air_batch_file in air_batch_files]
                water_batch = [self.read_img(water_batch_file) for water_batch_file in water_batch_files]
                depth_batch = [self.read_depth(depth_batch_file) for depth_batch_file in depth_batch_files]

                air_batch_images = np.array(air_batch).astype(np.float32)
                water_batch_images = np.array(water_batch).astype(np.float32)
                depth_batch_images = np.expand_dims(depth_batch, axis=3)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                # Update D network
                _, d_summary_str = self.sess.run([d_optim, d_sum],
                                                 feed_dict={self.z: batch_z,
                                                            self.water_inputs: water_batch_images,
                                                            self.air_inputs: air_batch_images,
                                                            self.depth_inputs: depth_batch_images})
                writer.add_summary(d_summary_str, global_step=counter)

                # Update G network
                _, g_summary_str = self.sess.run([g_optim, g_sum],
                                                 feed_dict={self.z: batch_z,
                                                            self.air_inputs: air_batch_images,
                                                            self.depth_inputs: depth_batch_images})
                writer.add_summary(g_summary_str, global_step=counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, g_summary_str = self.sess.run([g_optim, g_sum],
                                                 feed_dict={self.z: batch_z,
                                                            self.air_inputs: air_batch_images,
                                                            self.depth_inputs: depth_batch_images})
                writer.add_summary(g_summary_str, global_step=counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.air_inputs: air_batch_images,
                                                   self.depth_inputs: depth_batch_images})
                errD_real = self.d_loss_real.eval({self.water_inputs: water_batch_images})
                errG = self.g_loss.eval({self.z: batch_z, self.air_inputs: air_batch_images,
                                         self.depth_inputs: depth_batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, water_batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                print(self.sess.run('wc_generator/g_atten/g_eta_r:0'))
                print(self.sess.run('wc_generator/g_atten/g_eta_g:0'))
                print(self.sess.run('wc_generator/g_atten/g_eta_b:0'))
                print(self.sess.run('wc_generator/g_vig/g_amp:0'))
                print(self.sess.run('wc_generator/g_vig/g_tol:0'))
                print(self.sess.run('wc_generator/g_eta_rr:0'))
                print(self.sess.run('wc_generator/g_eta_gg:0'))
                print(self.sess.run('wc_generator/g_eta_bb:0'))

                if np.mod(epoch, self.save_epoch) == 0 and epoch != 0:
                    self.save(config.checkpoint_dir, counter)
                    print("Saving checkpoints")

    def test(self, config):
        """
        Test WGAN
        :param config: parameters config
        :return:
        """
        air_data = sorted(glob(os.path.join("./data", config.air_dataset, self.input_fname_pattern[0])))
        depth_data = sorted(glob(os.path.join("./data", config.depth_dataset, "*.mat")))
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        test_batch_size = self.batch_size
        sample_batch_idxs = len(air_data) // test_batch_size
        print(sample_batch_idxs)
        for idx in xrange(0, sample_batch_idxs):
            sample_air_batch_files = air_data[idx * test_batch_size:(idx + 1) * test_batch_size]
            sample_depth_batch_files = depth_data[idx * test_batch_size:(idx + 1) * test_batch_size]

            sample_air_batch = [self.read_img(sample_air_batch_file) for sample_air_batch_file in
                                sample_air_batch_files]
            sample_depth_batch = [self.read_depth(sample_depth_batch_file) for sample_depth_batch_file in
                                  sample_depth_batch_files]

            sample_air_images = np.array(sample_air_batch).astype(np.float32)
            sample_depth_images = np.expand_dims(sample_depth_batch, axis=3)
            sample_z = np.random.uniform(-1, 1, [test_batch_size, self.z_dim]).astype(np.float32)
            samples = self.sess.run([self.wc_sampler],
                                    feed_dict={self.z: sample_z,
                                               self.air_inputs: sample_air_images,
                                               self.depth_inputs: sample_depth_images})
            sample_ims = np.asarray(samples).astype(np.float32)
            sample_ims = np.squeeze(sample_ims)

            for img_idx in range(0, test_batch_size):
                out_file = "/fake_%06d_%03d.png" % (idx, img_idx)
                out_name = self.results_dir + out_file
                print(out_name)
                sample_im = sample_ims[img_idx, 0:self.output_height, 0:self.output_width, 0:3]
                sample_im = np.squeeze(sample_im)
                # sample_im = scipy.misc.imresize(sample_im, (self.sh, self.sw, 3), 'cubic')
                try:
                    scipy.misc.imsave(out_name, sample_im)
                    # cv2.imwrite(out_name, sample_im)
                except OSError:
                    print(out_name)
                    print("ERROR!")
                    pass
                out_file2 = "/air_%06d_%03d.png" % (idx, img_idx)
                out_name2 = self.results_dir + out_file2
                sample_im2 = sample_air_images[img_idx, 0:self.output_height, 0:self.output_width, 0:3]
                sample_im2 = np.squeeze(sample_im2)
                # sample_im2 = scipy.misc.imresize(sample_im2, (self.sh, self.sw, 3), 'cubic')
                try:
                    scipy.misc.imsave(out_name2, sample_im2)
                except OSError:
                    print(out_name)
                    print("ERROR!")
                    pass
                out_file3 = "/depth_%06d_%03d.mat" % (idx, img_idx)
                out_name3 = self.results_dir + out_file3
                sample_im3 = sample_depth_images[img_idx, 0:self.output_height, 0:self.output_width, 0]
                sample_im3 = np.squeeze(sample_im3)
                # sample_im3 = scipy.misc.imresize(sample_im3, (self.sh, self.sw), 'cubic', mode='F')
                try:
                    sio.savemat(out_name3, {'depth': sample_im3})
                except OSError:
                    print(out_name)
                    print("ERROR!")
                    pass

    def discriminator(self, image, reuse=False):
        """
        Discriminator for water_image and synthetic image
        :param image: input image
        :param reuse: Default as False
        :return: tf.nn.sigmoid
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.dfc_dim, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def sample_discriminator(self, image, reuse=False):
        """
        Discriminator for water_image and synthetic image
        :param image: input image
        :param reuse: Default as False
        :return: tf.nn.sigmoid
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.dfc_dim, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4)

    def wc_generator(self, z, image, depth):
        """
        JM generator model
        :param z: input noise
        :param image: input clear image
        :param depth: input clear image depth map
        :return: downgraded image
        """
        with tf.variable_scope("wc_generator"):

            # water-based attenuation and backscatter
            with tf.variable_scope("g_atten"):
                eta_r = tf.get_variable(name='g_eta_r', shape=[1, 1, 1], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0.35, stddev=0.01))
                eta_g = tf.get_variable(name='g_eta_g', shape=[1, 1, 1], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0.015, stddev=0.01))
                eta_b = tf.get_variable(name='g_eta_b', shape=[1, 1, 1], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0.036, stddev=0.01))

                eta = tf.stack([eta_r, eta_g, eta_b], axis=3)
                print('Attentuation: ', end=' ')
                print(eta)
                # Transmission map
                eta_d = tf.exp(tf.multiply(-1.0, tf.multiply(depth, eta)))
                print("attenuation Transmission map: ", end=' ')
                print(eta_d)

            # Direct Attenuation
            h0 = tf.multiply(image, eta_d)
            print("Direct Attenuation image: ", end=' ')
            print(h0)

            # backscatter
            z_, _, _ = linear(z, self.output_width * self.output_height*self.batch_size * 1, 'g_h0_lin', with_w=True)

            h0_z = tf.reshape(z_, [-1, self.output_height, self.output_width, self.batch_size * 1])
            h0_z = tf.nn.relu(self.g_bn0(h0_z))
            h0_z = tf.multiply(h0_z, depth)
            print("BackScatter Transmission map: ", end=' ')
            print(h0_z)

            with tf.variable_scope("g_vig"):
                A = tf.get_variable('g_amp', [1], initializer=tf.random_normal_initializer(mean=0.75, stddev=0.01))
                B = tf.get_variable('g_tol', [1], initializer=tf.random_normal_initializer(mean=0.9, stddev=0.01))

            with tf.variable_scope('g_h1_convg'):
                w = tf.get_variable('g_wg', [5, 5, h0_z.get_shape()[-1], 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1_zg = tf.nn.conv2d(h0_z, w, strides=[1, 1, 1, 1], padding='SAME')
            h_g = lrelu(self.g_bn1(h1_zg))

            with tf.variable_scope('g_h1_convr'):
                wr = tf.get_variable('g_wr', [5, 5, h0_z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1_zr = tf.nn.conv2d(h0_z, wr, strides=[1, 1, 1, 1], padding='SAME')
            h_r = lrelu(self.g_bn3(h1_zr))

            with tf.variable_scope('g_h1_convb'):
                wb = tf.get_variable('g_wb', [5, 5, h0_z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1_zb = tf.nn.conv2d(h0_z, wb, strides=[1, 1, 1, 1], padding='SAME')
            h_b = lrelu(self.g_bn4(h1_zb))

            h_r = tf.squeeze(h_r, axis=3)
            h_g = tf.squeeze(h_g, axis=3)
            h_b = tf.squeeze(h_b, axis=3)
            print("bscat_r: ", end=' ')
            print(h_r)
            print("bscat_g: ", end=' ')
            print(h_g)
            print("bscat_b: ", end=' ')
            print(h_b)

            # BackScattering air light
            bs_final0 = tf.stack([h_r, h_g, h_b], axis=3)
            # bs_final1 = tf.multiply(bs_final0, eta_d)
            # bs_final2 = tf.multiply(bs_final1, A)
            print("BackScatter image: ", end=' ')
            print(bs_final0)

            h2 = tf.add(bs_final0, h0)

            # Haze effect
            eta_rr = tf.constant(value=0.75, dtype=tf.float32, shape=[1, 1, 1], name='g_eta_rr')
            eta_gg = tf.constant(value=0.75, dtype=tf.float32, shape=[1, 1, 1], name='g_eta_gg')
            eta_bb = tf.constant(value=0.75, dtype=tf.float32, shape=[1, 1, 1], name='g_eta_bb')

            eta_haze = tf.stack([eta_rr, eta_gg, eta_bb], axis=3)
            tm_haze = tf.exp(tf.multiply(-1.0, tf.multiply(eta_haze, depth)))
            print("haze tm: ", end=' ')
            print(tm_haze)

            image_haze = tf.multiply(tf.multiply(255.0 * A, tf.subtract(1.0, tm_haze)), eta_d)
            print("haze image: ", end=' ')
            print(image_haze)

            # degrade image
            h_out = tf.add(h2, image_haze)
            h_out = tf.multiply(h_out, B)
            print("Degraded image: ", end=' ')
            print(h_out)

            return h_out, eta_r, eta_g, eta_b, A, B

    def wc_sample(self, z, image, depth):
        """
        JM generator model
        :param z: input noise
        :param image: air image
        :param depth: air image depth map
        :return: synthetic image
        """
        with tf.variable_scope("wc_generator", reuse=True):

            # water-based attenuation and backscatter
            with tf.variable_scope("g_atten", reuse=True):
                eta_r = tf.get_variable(name='g_eta_r', shape=[1, 1, 1], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0.35, stddev=0.01))
                eta_g = tf.get_variable(name='g_eta_g', shape=[1, 1, 1], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0.015, stddev=0.01))
                eta_b = tf.get_variable(name='g_eta_b', shape=[1, 1, 1], dtype=tf.float32,
                                        initializer=tf.random_normal_initializer(mean=0.036, stddev=0.01))

                eta = tf.stack([eta_r, eta_g, eta_b], axis=3)
                # Transmission map
                eta_d = tf.exp(tf.multiply(-1.0, tf.multiply(depth, eta)))

            # Direct Attenuation
            h0 = tf.multiply(image, eta_d)

            # backscatter
            z_, _, _ = linear(z, self.output_width * self.output_height*self.batch_size * 1, 'g_h0_lin', with_w=True)

            h0_z = tf.reshape(z_, [-1, self.output_height, self.output_width, self.batch_size * 1])
            h0_z = tf.nn.relu(self.g_bn0(h0_z))
            h0_z = tf.multiply(h0_z, depth)

            with tf.variable_scope("g_vig", reuse=True):
                A = tf.get_variable('g_amp', [1], initializer=tf.random_normal_initializer(mean=0.75, stddev=0.01))
                B = tf.get_variable('g_tol', [1], initializer=tf.random_normal_initializer(mean=0.9, stddev=0.01))

            with tf.variable_scope('g_h1_convg', reuse=True):
                w = tf.get_variable('g_wg', [5, 5, h0_z.get_shape()[-1], 1],
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1_zg = tf.nn.conv2d(h0_z, w, strides=[1, 1, 1, 1], padding='SAME')
            h_g = lrelu(self.g_bn1(h1_zg))

            with tf.variable_scope('g_h1_convr', reuse=True):
                wr = tf.get_variable('g_wr', [5, 5, h0_z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1_zr = tf.nn.conv2d(h0_z, wr, strides=[1, 1, 1, 1], padding='SAME')
            h_r = lrelu(self.g_bn3(h1_zr))

            with tf.variable_scope('g_h1_convb', reuse=True):
                wb = tf.get_variable('g_wb', [5, 5, h0_z.get_shape()[-1], 1],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            h1_zb = tf.nn.conv2d(h0_z, wb, strides=[1, 1, 1, 1], padding='SAME')
            h_b = lrelu(self.g_bn4(h1_zb))

            h_r = tf.squeeze(h_r, axis=3)
            h_g = tf.squeeze(h_g, axis=3)
            h_b = tf.squeeze(h_b, axis=3)

            # BackScattering air light
            bs_final0 = tf.stack([h_r, h_g, h_b], axis=3)
            # bs_final1 = tf.multiply(bs_final0, eta_d)
            # bs_final2 = tf.multiply(bs_final1, A)

            h2 = tf.add(bs_final0, h0)

            # Haze effect
            eta_rr = tf.constant(value=0.75, dtype=tf.float32, shape=[1, 1, 1], name='g_eta_rr')
            eta_gg = tf.constant(value=0.75, dtype=tf.float32, shape=[1, 1, 1], name='g_eta_gg')
            eta_bb = tf.constant(value=0.75, dtype=tf.float32, shape=[1, 1, 1], name='g_eta_bb')
            eta_haze = tf.stack([eta_rr, eta_gg, eta_bb], axis=3)
            tm_haze = tf.exp(tf.multiply(-1.0, tf.multiply(eta_haze, depth)))
            image_haze = tf.multiply(tf.multiply(255.0 * A, tf.subtract(1.0, tm_haze)), eta_d)

            # degrade image
            h_out = tf.add(h2, image_haze)
            h_out = tf.multiply(h_out, B)
            # h_out = tf.multiply(h_out1, A)

            return h_out

    @property
    def model_dir(self):
        """
        model dir
        :return:
        """
        return "{}_{}_{}_{}".format(
            self.water_dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        """
        save model to checkpoint path
        :param checkpoint_dir: checkpoint path
        :param step: tran step
        :return:
        """
        model_name = "WGan.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        """
        load checkpoint from checkpoint path....
        :param checkpoint_dir: checkpoint path
        :return:
        """
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False

    def read_depth(self, filename):
        """
        read depth map from file
        :param filename: depth map filename
        :return:
        """
        depth_mat = sio.loadmat(filename)
        depthtmp = depth_mat["dph"]
        # print(depthtmp.shape)
        if self.is_crop:
            depth = scipy.misc.imresize(depthtmp, (self.output_height, self.output_width), mode='F')
            depth = np.array(depth).astype(np.float32)
            #normalization
            # depth = np.divide(depth, depth.max())
            # print(depth.max())
            # print(depth.shape)
            # print(depth.max())
            # depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))
            # depth = np.divide(depth, depth.max())

            return depth
        else:
            depth = np.array(depthtmp).astype(np.float32)
            # normalization
            # depth = np.divide(depth, depth.max())
            # depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))
            # depth = np.divide(depth, depth.max())

            return depth

    def read_img(self, filename):
        """
        read img from file
        :param filename: img file name
        :return:
        """
        imgtmp = scipy.misc.imread(filename)
        # print(imgtmp.shape)

        if self.is_crop:
            img = scipy.misc.imresize(imgtmp, (self.output_height, self.output_width, 3))
            img = np.array(img).astype(np.float32)
            # normalize to [0, 1]
            # img = np.divide(img, 255)
            # print(img.max())
            return img
        else:
            img = np.array(imgtmp).astype(np.float32)
            # normalize to [0, 1]
            # img = np.divide(img, 255)
            return img

    def read_img_tm(self, filename):
        """
        estimate TM using DCP from water image
        :param filename: img file name
        :return:
        """
        imgtmp = scipy.misc.imread(filename)

        if self.is_crop:
            img = scipy.misc.imresize(imgtmp, (self.output_height, self.output_width, 3)) # (48, 64, 3)
            # img_tm = getTmap(img, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001) # (48, 64)
            # img = np.concatenate([img, img_tm], axis=-1)
            img = np.array(img).astype(np.float32)
            return img
        else:
            # img_tm = getTmap(imgtmp, omega=0.95, t0=0.1, blockSize=15, meanMode=False, percent=0.001)
            # img = np.concatenate([imgtmp, img_tm], axis=-1)
            img = np.array(imgtmp).astype(np.float32)
            return img

    def read_depth_small(self, filename):
        """
        read depth map file
        :param filename: depth map filename
        :return:
        """
        depth_mat = sio.loadmat(filename)
        depthtmp = depth_mat["dph"]
        # print(depthtmp.shape)

        if self.is_crop:
            depth = scipy.misc.imresize(depthtmp, (self.output_height, self.output_width), mode='F')
            depth = np.array(depth).astype(np.float32)
            # depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))
            # depth = np.divide(depth, depth.max())

            return depth
        else:
            depth = np.array(depthtmp).astype(np.float32)
            # depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))
            # depth = np.divide(depth, depth.max())
            return depth

    def read_depth_sample(self, filename):
        """
        read depth map from file
        :param filename: depth map filename
        :return:
        """
        depth_mat = sio.loadmat(filename)
        depthtmp = depth_mat["dph"]
        # print(depthtmp.shape)
        if self.is_crop:
            depth = scipy.misc.imresize(depthtmp, (self.sh, self.sw), mode='F')
            depth = np.array(depth).astype(np.float32)
            # depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))
            # depth = np.divide(depth, depth.max())

            return depth
        else:
            depth = np.array(depthtmp).astype(np.float32)
            # depth = np.multiply(self.max_depth, np.divide(depth, depth.max()))
            # depth = np.divide(depth, depth.max())

            return depth

    def read_img_sample(self, filename):
        """
        read image from file
        :param filename: image filename
        :return:
        """
        imgtmp = scipy.misc.imread(filename)
        # print(imgtmp.shape)
        if self.is_crop:
            img = scipy.misc.imresize(imgtmp, (self.sh, self.sw, 3))
            img = np.array(img).astype(np.float32)
            return img
        else:
            img = np.array(imgtmp).astype(np.float32)
            return img

