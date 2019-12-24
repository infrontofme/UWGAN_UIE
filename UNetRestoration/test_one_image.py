"""
U-Net implementation of test phase

Evaluation mode or Test mode

"""

import tensorflow as tf
import numpy as np
from scipy import misc
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from unet_model import UNet


image_path = './test/test_one/003744.jpg'
gt_image_path = './test/pro_003744.jpg'
ckpt_path = './checkpoints/'


def normalize_image(x):
    """
    [0, 255] -> [-1, 1]

    :param x: input image
    :return: normalized image
    """

    return (x / 127.5) - 1.0


if __name__ == '__main__':
    # underwater image
    image_u = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3], name='image_u')
    # correct image
    image_r = tf.placeholder(dtype=tf.float32, shape=[1, 256, 256, 3], name='image_r')
    # load test image
    test_image = normalize_image(misc.imresize(misc.imread(image_path), size=(256, 256), interp='cubic'))
    print(test_image)
    real_image = normalize_image(misc.imresize(misc.imread(gt_image_path), size=(256, 256), interp='cubic'))
    print(real_image)
    test_image_np = np.empty(shape=[1, 256, 256, 3], dtype=np.float32)
    test_image_np[0, :, :, :] = test_image
    test_image_tf = tf.convert_to_tensor(test_image_np)
    # laod model
    U_NET = UNet(input_=image_u, real_=image_r, is_training=False)
    gen_image = U_NET.u_net(inputs=test_image_tf, training=False)
    # load weight
    saver = tf.train.Saver(max_to_keep=1)
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    # generating
    begin_time = time.time()
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        # gen_image = U_NET.u_net(inputs=test_image, training=False)
        gen = np.asarray(sess.run(gen_image), dtype=np.float32)
        print(gen.shape)
        print(gen[0])
    end_time = time.time()
    print("Time cost: %f" % (end_time - begin_time))

    misc.imsave('./res_gen2.png', gen[0])
    misc.imsave('./img_ori2.png', test_image)
    misc.imsave('./img_real.png', real_image)
