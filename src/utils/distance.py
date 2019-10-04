import numpy as np


def euclidean_distance(x, y):
    """
    Computes the euclidean distance between two points.
    :param x: first point
    :param y: second point
    :return: euclidean distance between the points
    """
    return np.sqrt(np.sum(np.abs(x-y)**2))


def l1_distance(x, y):
    """
    Computes the absolute distance between two points.
    :param x: first point
    :param y: second point
    :return: absolute distance between the points
    """
    return np.sum(np.abs(x - y))
