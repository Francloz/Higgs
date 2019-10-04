import numpy as np
from numpy.linalg import inv
from src.utils.error import mse
from src.utils.data_manipulation import batch_iter


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    :param y: target
    :param tx: input
    :param initial_w: initial parameters
    :param max_iters: maximum iterations
    :param gamma: momentum
    :return: parameters, error
    """
    w = initial_w
    for it in range(max_iters):
        w = w - gamma/y.size * tx*(y-tx*w)
    return w, mse(y, tx*w)


def least_squares_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """
    Linear regression using stochastic gradient descent.

    :param y:
    :param tx: 
    :param initial_w: 
    :param max_iters: 
    :param gamma:
    :param batch_size: size of the batches
    :return: 
    """
    w = initial_w
    for it in range(max_iters):
        for yb, txb in batch_iter(y, tx, batch_size=batch_size):
            w = w - gamma/yb.size * txb*(yb-txb*w)
    return w, mse(y, tx*w)


def least_squares(y, tx):
    """"
    Least squares regression using normal equations.

    It assumes that tx*x is invertible.

    :param y:
    :param tx:
    :return: (Xt*X)^−1*Xt*y
    """
    return np.dot(np.dot(inv(np.dot(tx, np.transpose(tx, (1, 0)))), tx), y)
