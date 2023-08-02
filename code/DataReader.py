import os
import pickle
import numpy as np

cifar_dir = '../cifar-10-batches-py'
cifar_batch_size = 10000
num_features = 3072

def load_training_data():
    x_train = np.empty((5 * cifar_batch_size, num_features))
    y_train = np.empty((5 * cifar_batch_size, ))
    for i in range(1, 6):
        l = (i - 1) * cifar_batch_size
        r = i * cifar_batch_size
        x_train[l:r, :], y_train[l:r, ] = data_batch(cifar_dir + "/" + "data_batch_" + str(i))
    return x_train, y_train

def load_public_test_data():
    x_test = np.empty((cifar_batch_size, num_features))
    y_test = np.empty((cifar_batch_size, ))
    x_test[0:cifar_batch_size, :], y_test[0:cifar_batch_size, ] = data_batch(cifar_dir + "/" + "test_batch")
    return x_test, y_test

def load_private_test_data():
    # Returns (N, 3072)
    return np.load('../private_test_images_2022.npy').astype(np.float64)

def unpickle(file_path):
    with open(file_path, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')

def data_batch(file_path):
    data_batch = unpickle(file_path)
    return data_batch['data'], np.asarray(data_batch['labels'])