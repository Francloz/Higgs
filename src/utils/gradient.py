import numpy as np


def linear_gradient_mse(y, tx, w):
    """
    Computes the gradient of the error.
    The target and input must match the valid sample conditions.

    :param y: target
    :param tx: input
    :param w: parameters of the linear regression
    """
    return -1/y.size*tx.dot(y - tx*w)
