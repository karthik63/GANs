import scipy.misc
import numpy as np
import pickle
import gzip


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
