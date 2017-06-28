import scipy.misc
import numpy as np
import pickle
import pandas as pd
import gzip
import cv2
from math import ceil, floor
from PIL import Image

np.random.seed(12345)


class MNISTUtils:
    def __init__(self, path):
        f = gzip.open(path, 'rb')
        self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
        f.close()

        self.X_train_set = self.train_set[0]
        self.Y_category_n = self.train_set[1]
        self.X_valid_set = self.valid_set[0]
        self.Y_valid_set_n = self.valid_set[1]
        self.X_test_set = self.test_set[0]
        self.Y_test_set_n = self.test_set[1]

        self.Y_category_v = self.convert_number_to_vector(self.Y_category_n)
        self.Y_valid_set_v = self.convert_number_to_vector(self.Y_valid_set_n)
        self.Y_test_set_v = self.convert_number_to_vector(self.Y_test_set_n)

    @staticmethod
    def convert_number_to_vector(number_array):
        n = number_array.shape[0]
        vector_of_number = np.zeros((n, 10))

        for i in range(0, n):
            vector_of_number[i][number_array[i]] = 1

        return vector_of_number

    def get_label(self, index, label_path=None):
        label_vector = self.Y_category_v[index]
        return label_vector

    def get_image(self, index, image_path=None):
        image_vector = np.reshape(self.X_train_set[index], [28, 28, 1])
        return image_vector


class FashionUtils:
    def __init__(self, path, config):
        self.img_height = config.input_height
        self.img_width = config.input_width
        self.resize = config.resize

        self.path = path
        raw_file = open(path + '/Anno/list_category_img.txt', 'rb').read().decode()

        raw_file = raw_file.split('\n')[2:-1]
        raw_file = [k.split() for k in raw_file]
        self.image_categories = pd.DataFrame([[k[0], int(k[1])] for k in raw_file])

        raw_file = open(path + '/Anno/list_bbox.txt', 'rb').read().decode()

        raw_file = raw_file.split('\n')[2:-1]
        raw_file = [k.split() for k in raw_file]
        self.image_bboxes = pd.DataFrame([[k[0], int(k[1]), int(k[2]), int(k[3]), int(k[4])] for k in raw_file])

        self.image_info = pd.concat(
            [self.image_categories, self.image_bboxes[[1, 2, 3, 4]].rename(columns={1: 2, 2: 3, 3: 4, 4: 5})], axis=1)

        self.n_images = self.image_categories.shape[0]
        self.bbox = np.zeros([self.n_images, 4], dtype=int)

        self.image_info = self.image_info.sample(frac=1).reset_index(drop=True)

        self.Y_category_n = np.array(self.image_info[1], dtype='int')

        self.Y_category_v = self.convert_number_to_vector(self.Y_category_n)

        self.bbox = np.array(self.image_info[[2, 3, 4, 5]], dtype='int')

        self.image_categories = self.image_info[0]

        pickle.dump(self.Y_category_n, open('Y_category_n', 'wb'))
        pickle.dump(self.Y_category_v, open('Y_category_v', 'wb'))

    def get_image(self, index):
        img = cv2.imread(self.path + '/' + self.image_categories[index]).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img /= 255
        img = self.crop_bbox(img, index)

        img_h, img_w, img_channels = img.shape
        top, bottom = int(floor((299.0 - img_h) / 2)), int(ceil((299.0 - img_h) / 2))
        left, right = int(floor((299.0 - img_w) / 2)), int(ceil((299.0 - img_w) / 2))

        if not self.resize:
            modified_img = cv2.copyMakeBorder(img, top, bottom, left, right, value=[0, 0, 0],
                                                                borderType=cv2.BORDER_CONSTANT)
        else:
            modified_img = cv2.resize(img, (self.img_width, self.img_height))

        return modified_img[:, :, :]

    def get_label(self, index):
        return self.Y_category_v[index]

    def crop_bbox(self, img, index):
        left = self.bbox[index][0]
        top = self.bbox[index][1]
        right = self.bbox[index][2]
        bottom = self.bbox[index][3]

        return img[top:bottom, left:right, :]

    @staticmethod
    def convert_number_to_vector(number_array):
        n = number_array.shape[0]
        vector_of_number = np.zeros((n, 50))

        for i in range(0, n):
            vector_of_number[i][number_array[i]] = 1

        return vector_of_number