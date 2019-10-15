# -*- coding: utf-8 -*-
"""
Some functions might be copied from:
https://github.com/epfml/ML_course/blob/master/labs/ex02/template/helpers.py
"""
import numpy as np


def separate(dataset, test=.3):
    """
    Separates the dataset into training and testing batches.
    :param dataset:
    :param test: relative sizee eof the testing batch
    :return:
    """
    pass


def load_dataset(path):
    """
    Load dataset from file
    :param path: path to the file
    :return: dataset found in the file
    """
    return np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1,
                      converters={1: lambda x: 1 if b"b" in x else 0})


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]