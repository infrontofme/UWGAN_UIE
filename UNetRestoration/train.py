"""
Main training file

The goal is to correct the colors in underwater images.
The image pair contains color-distort image (which can be generate by CycleGan),and ground-truth image
Then, we use the u-net, which will attempt to correct the colors

"""

import tensorflow as tf
from scipy import misc
import math
import glob
import numpy as np
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# my imports
from unet_model import UNet
from utils import BatchRename, ImageProcess


init_learning_rate = 1e-4

# Momentum Optimizer
nesterov_momentum = 0.9

# l2 regularizer
weight_decay = 1e-4

batch_size = 32
total_epochs = 10


trainA_path = '/home/frost/image_enhance/UIE/UWGAN_Results/Water_near_1/results_1/water/'
trainB_path = '/home/frost/image_enhance/UIE/UWGAN_Results/Water_near_1/results_1/air/'
log_path = './Far2/logs_l1/'
ckpt_path = './Far2/checkpoints_l1/'


def cosine_learning_rate(learn_rate, n_epochs, cur_epoch):
    """
    cosine decay learning rate from 0.1~0, during training phase
    :param learn_rate: 0.1, initial learning rate
    :param n_epochs: 300, total epochs
    :param epoch: current epoch
    :return: cosine_learning_rate
    """
    t_total = n_epochs
    t_cur = cur_epoch
    learning_rate_cosine = 0.5 * learn_rate * (1 + math.cos(math.pi * t_cur / t_total))

    return learning_rate_cosine


if __name__ == "__main__":

    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    print("Params Config:\n")
    print("Learning Rate: %f" % init_learning_rate)
    print("    Optimizer: Adam")
    print("   Batch Size: %d " % batch_size)
    print(" Train Epochs: %d " % total_epochs)

    # rename pic for underwater image and ground truth image
    # BatchRename(image_path=trainA_path).rename()
    # BatchRename(image_path=trainB_path).rename()

    # underwater image
    image_u = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_u')

    # correct image
    image_r = tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 3], name='image_r')

    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
    lr_sum = tf.summary.scalar('lr', learning_rate)

    # generated color image by u-net
    U_NET = UNet(input_=image_u, real_=image_r, is_training=training_flag)
    gen_image = U_NET.u_net(inputs=image_u, training=training_flag)
    G_sum = tf.summary.image("gen_image", gen_image, max_outputs=10)

    # loss of u-net
    errG = U_NET.l1_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.mse_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.ssim_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.msssim_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.gdl_loss(gt=image_r, gen=gen_image)
    # errG = U_NET.l2_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)
    # errG = U_NET.ssim_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)
    # errG = U_NET.msssim_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)
    # errG = U_NET.gdl_l1_loss(gt=image_r, gen=gen_image, alpha=0.8)

    errG_sum = tf.summary.scalar("loss", errG)
    t_var = tf.trainable_variables()
    g_vars = [var for var in t_var]

    # if consider l2 regularization
    # l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in t_var])

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
    # train_op = optimizer.minimize(errG + l2_loss * weight_decay)
    train_op = optimizer.minimize(loss=errG)

    # TensorBoard Summaries
    # tf.summary.scalar('batch_loss', tf.reduce_mean(errG))
    # tf.summary.scalar('learning_rate', learning_rate)
    # try:
    #     tf.summary.scalar('l2_loss', tf.reduce_mean(l2_loss))
    # except: pass

    # saver = tf.train.Saver(tf.global_variables())

    config = tf.ConfigProto()
    # restrict model GPU memory utilization to min required
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        ckpt = tf.train.get_checkpoint_state(ckpt_path)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            U_NET.restore(sess=sess, model_path=ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        all_sum = tf.summary.merge([G_sum, errG_sum, lr_sum])
        train_summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        # merged_summary_op = tf.summary.merge_all()

        # load data
        # trainA_paths for underwater image
        # trainB_paths for ground truth image
        img_process = ImageProcess(pathA=trainA_path + '*.png',
                                   pathB=trainB_path + '*.png',
                                   batch_size=batch_size,
                                   is_aug=False)
        counter = 1
        trainA_paths, trainB_paths = img_process.load_data()
        for epoch in range(1, total_epochs+1):
            # epoch_learning_rate = cosine_learning_rate(learn_rate=init_learning_rate,
            #                                            n_epochs=total_epochs,
            #                                            cur_epoch=epoch)
            epoch_learning_rate = init_learning_rate
            # total_loss = []
            start_time = time.time()
            for step in range(1, int(len(trainA_paths)/batch_size)):

                batchA_images, batchB_images = img_process.shuffle_data(trainA_paths, trainB_paths)

                train_feed_dict = {
                    image_u: batchA_images,
                    image_r: batchB_images,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, summary_str = sess.run([train_op, all_sum], feed_dict=train_feed_dict)
                train_summary_writer.add_summary(summary=summary_str, global_step=counter)

                # batch_loss = sess.run(errG, feed_dict=train_feed_dict)
                # total_loss.append(batch_loss)
                counter += 1

            end_time = time.time()
            # train_loss = np.mean(total_loss)
            line = "epoch: %d/%d, time cost: %.4f\n" % (epoch, total_epochs, float(end_time - start_time))
            # line = "epoch: %d/%d, train loss: %.4f, time cost: %.4f\n" % (epoch, total_epochs, float(train_loss), float(end_time - start_time))
            print(line)

            if epoch % 10 == 0:
                U_NET.save(sess=sess, model_path=ckpt_path + str(epoch)+'u_net.ckpt')


