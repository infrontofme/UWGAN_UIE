"""
U-Net implementation of test phase

Evaluation mode or Test mode

"""

import tensorflow as tf
import numpy as np
from scipy import misc
import glob
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from unet_model import UNet


test_path = '/media/wangnan/Data/UIE/data/Synthetic_water_near/'
gen_path = '/media/wangnan/Data/UIE/Unet_res/Res/3-Near/gen_gdll1_synwaternear/'
ckpt_path = '/media/wangnan/Data/UIE/Unet_ckpt/Near/checkpoints_gdll1/'
test_batch_size = 1


def normalize_image(x):
    """
    [0, 255] -> [-1, 1]

    :param x: input image
    :return: normalized image
    """

    # [0, 255] -> [0, 1]

    # return cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX, -1)

    return x / 255.0

    # return (x / 127.5) - 1.0


if __name__ == '__main__':

    if not os.path.exists(gen_path):
        os.makedirs(gen_path)
    print("Params Config:\n")
    print("   Batch Size: %f" % test_batch_size)

    # underwater image
    image_u = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_u')
    # correct image
    image_r = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_r')

    # load model
    U_NET = UNet(input_=image_u, real_=image_r, is_training=False)
    gen_images = U_NET.u_net(inputs=image_u, training=False)

    # load image
    test_path = np.asarray(glob.glob(test_path + '*.png'))
    test_path.sort()
    # process image
    num_test_image = len(test_path)
    test_x = np.empty(shape=[num_test_image, 256, 256, 3], dtype=np.float32)
    i = 0
    for path_i in test_path:
        img_i = normalize_image(misc.imresize(misc.imread(path_i).astype('float32'), size=(256, 256), interp='cubic'))
        # img_i = normalize_image(misc.imread(path_i).astype('float32'))
        test_x[i, :, :, :] = img_i
        i += 1
    # load checkpoints weight
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # compute FLOPs and params###################################################################
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(tf.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(tf.Session().graph, run_meta=run_meta, cmd='op', options=opts)
        print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
        #############################################################################################

        U_NET.restore(sess, model_path=ckpt.model_checkpoint_path)

        test_pre_index = 0
        gen_num = 0
        gen_test_images = []
        begin_time = time.time()
        for test in range(int(num_test_image / test_batch_size)):
            test_batch_x = test_x[test_pre_index:test_pre_index+test_batch_size]
            test_pre_index = test_pre_index + test_batch_size

            test_feed_dict = {image_u: test_batch_x}

            gen_test = sess.run(gen_images, feed_dict=test_feed_dict)
            # print(gen_test.max())
            # print(gen_test.min())
            # print(gen_test.shape)
            gen_test_images.append(gen_test)
            # print(len(gen_test_images))
            # print(gen_test_images[0][0].shape)
        end_time = time.time()

        print("Total Test Time: %.4f, Average Time Per Image: %.6f, FPS: %.4f" %
              (end_time - begin_time,
               (end_time - begin_time) / num_test_image,
               num_test_image / (end_time - begin_time)))

        for img_g in gen_test_images:
            # misc.imsave(gen_path + str(gen_num) + '_gen.png', img_g[0])
            misc.imsave(gen_path + test_path[gen_num].split('/')[-1][:-4] + '_gen.png', img_g[0])
            # print(img_g[0].shape)
            gen_num += 1
        print("Done with test image, gen image: %d" % gen_num)