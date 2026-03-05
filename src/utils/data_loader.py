"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras import datasets

def load_mnist():
    '''
    shape of x_train: (60000,28,28)
    shape of y_train: (60000,)
    shape of x_test: (10000,28,28)
    shape of y_test: (10000,)
    '''
    (x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()
    #reshaping the input as the input layer is a 1D vector
    '''
    new shape of x_train: (60000,784)
    new shape of x_test: (10000,784)
    '''
    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))


    return x_train,y_train,x_test,y_test


def load_fashion_mnist():
    '''
    shape of x_train: (60000,28,28)
    shape of y_train: (60000,)
    shape of x_test: (10000,28,28)
    shape of y_test: (10000,)
    '''
    (x_train,y_train),(x_test,y_test) = datasets.fashion_mnist.load_data()
     #reshaping the input as the input layer is a 1D vector
    '''
    new shape of x_train: (60000,784)
    new shape of x_test: (10000,784)
    '''
    x_train = np.reshape(x_train,(60000,784))
    x_test = np.reshape(x_test,(10000,784))

    return x_train,y_train,x_test,y_test

