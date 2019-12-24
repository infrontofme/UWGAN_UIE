"""
U-Net implementation in TensorFlow

Objective: Underwater image restoration

y = f(x)

x: distorted image (512, 512, 3)
y: restored image (512, 512, 3)

Loss function:
    L = alpha * L ^(MS-SSIM) + (1 - alpha) * L ^(L1)

    Multi-scale Structural SIMilarity index and absolute value(L1) loss functions

Original Paper:
    https://arxiv.org/abs/1505.04597
    https://arxiv.org/abs/1905.09000

"""


import tensorflow as tf


def conv_conv_pool(input_, n_filters, training, name, pool=True):
    """
    {conv ->  BN -> Relu} x 2 -> {max-pooling}

    Args:
        :param input_: (4-D Tensor): (batch_size, H, W, C)
        :param n_filters: (int): number of filters
        :param training: (1-D Tensor): Boolean Tensor
        :param name: (str): name postfix
        :param pool: (bool): If True, MaxPool2D
        :return: (4-D Tensor): output of the operations
    """

    net = input_

    with tf.variable_scope("layer{}".format(name)):
        net = tf.layers.conv2d(inputs=net,
                               filters=n_filters,
                               kernel_size=(3, 3),
                               padding='SAME',
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name="conv_1")
        # net = tf.layers.batch_normalization(inputs=net,
        #                                     training=training,
        #                                     name="bn_1")
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(inputs=net,
                               filters=n_filters,
                               kernel_size=(3, 3),
                               padding='SAME',
                               kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                               name="conv_2")
        # net = tf.layers.batch_normalization(inputs=net,
        #                                     training=training,
        #                                     name="bn_2")
        net = tf.nn.relu(net)

        if pool is False:
            return net

        # if pool is True
        pool_net = tf.layers.max_pooling2d(inputs=net,
                                           pool_size=(2, 2),
                                           strides=2,
                                           name="pool_{}".format(name))

        return net, pool_net


def upconv_2d(input_, n_filters, name):
    """
    up convolution input tensor

    Args:
        :param input_: (4-D Tensor): (N, H, W, C)
        :param n_filters: (int): number of filters, filter size
        :param name: (str): name of up-sampling operations
        :return: output(4-D Tensor): (N, 2*H, 2*W, C/2)
    """

    net = tf.layers.conv2d_transpose(inputs=input_,
                                     filters=n_filters,
                                     kernel_size=2,
                                     strides=2,
                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                     name="upsample_{}".format(name))

    return net


def upconv_concat(input_A, input_B, n_filters, name):
    """
    Up-sample input_A and concat with input_B

    Args:
        :param input_A: (4-D Tensor): (N, W, H, C)
        :param input_B: (4-D Tensor): (N, 2*W, 2*H, C2)
        :param n_filters: (int): number of filters, filter size
        :param name: (str): name of the concat operations
        :return: output(4-D Tensor): (N, 2*H, 2*W, C+C2)
    """

    upconv = upconv_2d(input_A, n_filters, name)

    net = tf.concat(values=[upconv, input_B],
                    axis=-1,
                    name="concat_{}".format(name))

    return net


class UNet(object):
    """
    An u-net implementation

    """
    def __init__(self, input_, real_, is_training=True):
        """
        :param input_: input image
        :param real: ground truth of image
        :param is_training: bool whether or not is in training mode
        """
        self.input_ = input_
        self.real_ = real_
        self.is_training = is_training
        # self.output_ = self.u_net(inputs=self.input_, training=self.is_training)
        # self.cost = self.msssim_l1_loss(gt=self.real_, gen=self.output_, alpha=0.8)

    def u_net(self, inputs, training):
        """
        Build U-net Architecture

        Args:
            :param inputs: (4-D Tensor): (N, H, W, C), input image
            :param training: (1-D Tensor): Boolean Tensor is required for batch normalization layers
            :return: output(4-D Tensor): (N, H, W, C), same shape as inputs

        Notes:
            Underwater Denoising Autoencoder using U-Net
            https://arxiv.org/abs/1905.09000
        """

        # (N, 512, 512, 3)
        net = inputs
        # resize image to 256 * 256
        # net = tf.image.resize_images(images=net, size=(256, 256), method=tf.image.ResizeMethod.AREA)
        print(net)
        # Down Sample
        # (N, 512, 512, 32) \ (N, 256, 256, 32)
        conv1, pool1 = conv_conv_pool(input_=net, n_filters=32, training=training, name=1, pool=True)
        # (N, 256, 256, 64) \ (N, 128, 128, 64)
        conv2, pool2 = conv_conv_pool(input_=pool1, n_filters=64, training=training, name=2, pool=True)
        # (N, 128, 128, 128) \ (N, 64, 64, 128)
        conv3, pool3 = conv_conv_pool(input_=pool2, n_filters=128, training=training, name=3, pool=True)
        # (N, 64, 64, 256) \ (N, 32, 32, 256)
        conv4 = conv_conv_pool(input_=pool3, n_filters=256, training=training, name=4, pool=False)

        # Up Sample
        # (N, 128, 128, 256) \ (N, 64, 64, 256)
        up5 = upconv_concat(input_A=conv4, input_B=conv3, n_filters=128, name=5)
        # (N, 128, 128, 128) \ (N, 64, 64, 128)
        conv5 = conv_conv_pool(input_=up5, n_filters=128, training=training, name=5, pool=False)

        # (N, 256, 256, 128) \ (N, 128, 128, 128)
        up6 = upconv_concat(input_A=conv5, input_B=conv2, n_filters=64, name=6)
        # (N, 256, 256, 64) \ (N, 128, 128, 64)
        conv6 = conv_conv_pool(input_=up6, n_filters=64, training=training, name=6, pool=False)

        # (N, 512, 512, 64) \ (N, 256, 256, 64)
        up7 = upconv_concat(input_A=conv6, input_B=conv1, n_filters=32, name=7)
        # (N, 512, 512, 32) \ (N, 256, 256, 32)
        conv7 = conv_conv_pool(input_=up7, n_filters=32, training=training, name=7, pool=False)

        # 1x1 conv, (N, 512, 512, 3) \ (N, 256, 256, 3)
        output = tf.layers.conv2d(inputs=conv7,
                                  filters=3,
                                  kernel_size=(1, 1),
                                  padding='SAME',
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  activation=tf.nn.tanh)
        # resize to original size 405 x 720
        # output = tf.image.resize_images(images=output, size=(405, 720), method=tf.image.ResizeMethod.BICUBIC)
        print(output)

        return output

    """
    
    Loss functions for Underwater image restoration
    We referred to paper ”Loss functions for image restoration with neural networks“,
    where the loss function can be expressed as:
        L = alpha * L^(MS-SSIM) + (1 - alpha) * L^(L1)
        
        alpha was set to be 0.8 in paper "Underwater Color Restoration Using U-Net Denoising Autoencoder"
    
    """

    def l1_loss(self, gt, gen):
        """
         Absolute Difference loss between gt and gen.

        Args:
            :param gt: The ground truth output tensor, same dimensions as 'gen'.
            :param gen: The predicted outputs.
            :return: Weighted loss float Tensor, it is scalar.
        """
        return tf.losses.absolute_difference(labels=gt, predictions=gen, scope='l1_loss')

    def mse_loss(self, gt, gen):
        """
        l2 loss between gt and gen

        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :return: L2 loss
        """
        return tf.losses.mean_squared_error(
            labels=gt,
            predictions=gen)

    def ssim_loss(self, gt, gen):
        """
        Structural Similarity loss between gt and gen

        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :return: Loss
        """
        return 1 - tf.reduce_mean(
            tf.image.ssim(
                gen,
                gt,
                max_val=1))

    def msssim_loss(self, gt, gen):
        """
        Computes the MS-SSIM loss between gt and gen

        Args:
            :param gt: The ground truth output tensor, same dimensions as 'gen'.
            :param gen: The predicted outputs.
            :return: Weighted loss float Tensor, it is scalar.
        """
        return 1 - tf.reduce_mean(
            tf.image.ssim_multiscale(
                img1=gen,
                img2=gt,
                max_val=1)
        )

    def gdl_loss(self, gt, gen):
        """
        Compute the image gradient loss between gt and gen
        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :return: gdl_loss, it is scalar
        """
        dy_gt, dx_gt = tf.image.image_gradients(gt)
        dy_gen, dx_gen = tf.image.image_gradients(gen)
        grad_loss = tf.reduce_mean(tf.abs(dy_gen - dy_gt) + tf.abs(dx_gen - dx_gt))

        return grad_loss

    def l2_l1_loss(self, gt, gen, alpha=0.8):
        """
        Loss function mix l1_loss and l2_loss

        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :param alpha: coefficient, default set as 0.8
        :return: Loss
        """
        l1 = self.l1_loss(gt, gen)
        l2 = self.mse_loss(gt, gen)

        return alpha * l2 + (1 - alpha) * l1

    def ssim_l1_loss(self, gt, gen, alpha=0.8):
        """
        Loss function, calculating alpha * ssim_loss + (1-alpha) * l1_loss
        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :param alpha: coefficient, set to 0.8 according to paper
        :return: Loss
        """
        l1 = self.l1_loss(gt, gen)
        ssim_loss = self.ssim_loss(gt, gen)

        return alpha * ssim_loss + (1 - alpha) * l1

    def msssim_l1_loss(self, gt, gen, alpha=0.8):
        """
        Loss function, calculating alpha * msssim_loss + (1-alpha) * l1_loss
        according to 'Underwater Color Restoration Using U-Net Denoising Autoencoder' [Yousif]
        :alpha: default value accoording to paper

        Args:
            :param gt: The ground truth output tensor, same dimensions as 'gen'.
            :param gen: The predicted outputs.
            :param alpha: coefficient, set to 0.8 according to paper
            :return: Loss
        """
        l1 = self.l1_loss(gt, gen)

        # ssim_multiscale already calculates the dyalidic pyramid (with as replacment avg.pooling)
        msssim_loss = self.msssim_loss(gt, gen)

        return alpha * msssim_loss + (1 - alpha) * l1

    def gdl_l1_loss(self, gt, gen, alpha=0.8):
        """
        Loss function, calculating alpha * gdl_loss + (1-alpha) * l1_loss
        :param gt: The ground truth output tensor, same dimensions as 'gen'.
        :param gen: The predicted outputs.
        :param alpha: coefficient, set to 0.8 according to paper
        :return: Loss
        """
        l1 = self.l1_loss(gt, gen)
        gdl = self.gdl_loss(gt, gen)

        return alpha * gdl + (1 - alpha) * l1

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        :return: save_path
        """
        saver = tf.train.Saver(tf.global_variables())
        save_path = saver.save(sess, model_path)

        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        :return: None
        """

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, model_path)
        print("Model restored from file: %s" % model_path)

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data

        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [N, H, W, C]
        :return: prediction: The unet prediction, restored image shape [N, H, W, C]
        """

        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Restore model weight from previously saved model
            self.restore(sess, model_path)

            y_gen = self.u_net(x_test, training=False)

            return sess.run(y_gen)


# if __name__ == "__main__":
#     with tf.Session() as sess:
#         input_image = tf.placeholder("float", shape=[1, 512, 512, 3], name="input")
#         real_image = tf.placeholder("float", shape=[1, 512, 512, 3], name="real")
#
#         unet = UNet(input_=input_image, real_=real_image, is_training=True)
#
#         print(unet.cost)


