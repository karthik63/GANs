import scipy.misc
import numpy as np
import pickle
import gzip
import cv2
from math import ceil, floor
from PIL import Image

class MNISTUtils:
    def __init__(self, path):
        f = gzip.open(path, 'rb')
        self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
        f.close()

        self.X_train_set = self.train_set[0]
        self.Y_train_set_n = self.train_set[1]
        self.X_valid_set = self.valid_set[0]
        self.Y_valid_set_n = self.valid_set[1]
        self.X_test_set = self.test_set[0]
        self.Y_test_set_n = self.test_set[1]

        self.Y_train_set_v = self.convert_number_to_vector(self.Y_train_set_n)
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
        label_vector = self.Y_train_set_v[index]
        return label_vector

    def get_image(self, index, image_path = None):
        image_vector = np.reshape(self.X_train_set[index], [28, 28, 1])
        return image_vector

class FashionUtils:

    def __init__(self, path):

        self.path = path
        raw_file = open(path + '/Anno/list_category_img.txt', 'rb').read().decode()

        raw_file = raw_file.split('\n')[2:-1]
        raw_file = [k.split() for k in raw_file]
        self.image_categories = [[k[0], int(k[1])] for k in raw_file]

        self.n_images = len(self.image_categories)
        self.Y_train_set_n = np.zeros([self.n_images], dtype=int)

        for i in range(self.n_images):
            self.Y_train_set_n[i] = self.image_categories[i][1]

        self.Y_train_set_v = self.convert_number_to_vector(self.Y_train_set_n)

        pickle.dump(self.Y_train_set_n, open('Y_train_set_n', 'wb'))
        pickle.dump(self.Y_train_set_v, open('Y_train_set_v', 'wb'))

    def get_image(self, index):
        img = cv2.imread(self.path + '/' + self.image_categories[index][0]).astype(np.float32)
        img /= 255
        img_h, img_w, img_channels = img.shape
        top, bottom = int(floor((301 - img_h) / 2)), int(ceil((301 - img_h) / 2))
        left, right = int(floor((301 - img_w) / 2)), int(ceil((301 - img_w) / 2))
        img_with_border = cv2.copyMakeBorder(img, top, bottom, left, right, value=[0, 0, 0],
                                             borderType=cv2.BORDER_CONSTANT)
        return img_with_border

    def get_label(self, index):
        return self.Y_train_set_v[index]

    @staticmethod
    def convert_number_to_vector(number_array):
        n = number_array.shape[0]
        vector_of_number = np.zeros((n, 50))

        for i in range(0, n):
            vector_of_number[i][number_array[i]] = 1

        return vector_of_number