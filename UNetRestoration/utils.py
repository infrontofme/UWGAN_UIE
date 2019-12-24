"""
Process image before training

TrainA: distort image
TrainB: clean image

1. Normalization for TrainA and TrainB
2. shuffle TrainA and TrainB
3. Augmentation (optional)
"""

import os
import glob
import random
import numpy as np
# import cv2
from scipy import misc


class BatchRename(object):
    """
    Rename picture name in the folder

    """

    def __init__(self, image_path):
        """
        :param image_path: image path, where to store image
        """
        self.path = image_path

    def rename(self):

        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i).zfill(6) + '.jpg')
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                except Exception as e:
                    print("type error: " + str(e))
                i = i + 1
        print('total %d to rename & converted %d jpgs' % (total_num, i-1))


class ImageProcess():
    """
    Process image before training
    normalize, shuffle, augmentation (optional)

    """

    def __init__(self, pathA, pathB, batch_size, is_aug=True):

        # underwater photos
        self.trainA_path = pathA

        # normal photos (ground truth)
        self.trainB_path = pathB

        self.batch_size = batch_size
        self.is_aug = is_aug

    def load_data(self):

        """
        load training image

        :return: trainA and trainB path
        """

        trainA_paths = np.asarray(glob.glob(self.trainA_path))
        trainA_paths.sort()
        # print(trainA_paths[233])
        trainB_paths = np.asarray(glob.glob(self.trainB_path))
        trainB_paths.sort()
        # print(trainB_paths[233])

        print(len(trainB_paths), 'training images')

        return trainA_paths, trainB_paths

    def shuffle_data(self, trainA_paths, trainB_paths):
        """
        shuffle training image

        :return: shuffled data path
        """

        num_train = len(trainA_paths)
        idx = np.random.choice(np.arange(num_train), self.batch_size, replace=False)
        batchA_paths = trainA_paths[idx]
        # print(batchA_paths)
        batchB_paths = trainB_paths[idx]
        # print(batchB_paths)

        batchA_images = np.empty(shape=[self.batch_size, 256, 256, 3], dtype=np.float32)
        batchB_images = np.empty(shape=[self.batch_size, 256, 256, 3], dtype=np.float32)

        i = 0
        for a, b in zip(batchA_paths, batchB_paths):
            # a_img = self.normalize_image(misc.imresize(misc.imread(a).astype('float32'),
            #                                            size=(256, 256), interp='cubic'))
            a_img = self.normalize_image(misc.imread(a).astype('float32'))
            # b_img = self.normalize_image(misc.imresize(misc.imread(b).astype('float32'),
            #                                            size=(256, 256), interp='cubic'))
            b_img = self.normalize_image(misc.imread(b).astype('float32'))

            # data augmentation
            if self.is_aug:
                a_img, b_img = self.augmentation_image(a_img, b_img)

            batchA_images[i, ...] = a_img
            batchB_images[i, ...] = b_img
            i += 1

        return batchA_images, batchB_images

    def normalize_image(self, x):
        """
        [0, 255] -> [-1, 1]

        :param x: input image
        :return: normalized image
        """
        #[0, 255] -> [0, 1]

        # return cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX, -1)

        return x / 255.0

        # return (x / 127.5) - 1.0

    def augmentation_image(self, x, y):
        """
        Data Augmentation for image

        :param x: input image a
        :param y: input image b
        :return: augmented image
        """

        r = random.random()
        # flip image left right
        if r < 0.5:
            x = np.fliplr(x)
            y = np.fliplr(y)

        r = random.random()
        # flip image up down
        if r < 0.5:
            x = np.flipud(x)
            y = np.flipud(y)

        r = random.random()
        # send in the clean image for both
        if r < 0.5:
            x = y

        return x, y


# if __name__ == '__main__':
#
#     img = ImageProcess(pathA ='./dataset/trainA/*.jpg',
#                        pathB ='./dataset/trainB/*.jpg',
#                        batch_size=1,
#                        is_aug=True)
#     a, b = img.load_data()
#     im_a, im_b = img.shuffle_data(a, b)
#     print(a, b)
#     print(im_a, im_b)