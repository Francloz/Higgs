import numpy as np


def assert_vector(x):
    """
    Asserts that the input is a vector.
    :param x:
    """
    assert sum(i != 1 for i in x.shape) <= 1


def column(x):
    """
    Transforms the vector into a column.
    :param x: vector
    :return column numpy array
    """
    assert_vector(x)
    return x.reshape(-1, 1)


def row(x):
    """
    Transforms the vector into a row.
    :param x: vector
    :return row numpy array
    """
    assert_vector(x)
    return x.reshape(1, -1)


def assert_dimension(x,y):
    """
    Asserts that two vectors exist in the same space.
    :param x: first vector
    :param y: second vector
    """
    assert_vector(x)
    assert_vector(y)
    assert sum(i for i in x.shape if i > 1) == sum(i for i in y.shape if i > 1)


def assert_valid_sample(y, tx):
    """
    Asserts that the dimensions match and that the sample is transposed.
    :param y: target
    :param tx: transposed sample
    """
    assert_vector(y)
    assert_vector(tx[0])
    assert_dimension(y.shape[0], tx.shape[0])
