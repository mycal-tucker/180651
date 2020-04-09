from keras.datasets import mnist
import numpy as np


def get_data():
    # Load the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Shuffle the training data
    permutation = np.random.permutation(x_train.shape[0])
    x_train = x_train[permutation]
    y_train = y_train[permutation]
    # Shuffle the test data.
    permutation = np.random.permutation(x_test.shape[0])
    x_test = x_test[permutation]
    y_test = y_test[permutation]
    # Create the one-hot versions of y
    y_train_one_hot = np.zeros((x_train.shape[0], 10))
    y_test_one_hot = np.zeros((x_test.shape[0], 10))
    for i, y in enumerate(y_train):
        y_train_one_hot[i][y] = 1
    for i, y in enumerate(y_test):
        y_test_one_hot[i][y] = 1
    return x_train, y_train, y_train_one_hot, x_test, y_test, y_test_one_hot
