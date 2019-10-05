import numpy as np
from numpy.linalg import inv
from src.utils.error import mse
from src.utils.data_manipulation import batch_iter
from src.utils.error import *
from src.utils.scheduler import Scheduler


def least_squares_gd(y, tx, initial_w, max_iters, gamma, loss_function=mse, max_error=0):
    """
    Linear regression using gradient descent.

    :param y: label
    :param tx: input
    :param initial_w: initial parameters
    :param max_iters: maximum iterations
    :param gamma: gradient factor, it can be a Scheduler
    :param loss_function: loss function used to compute the loss
    :param max_error: maximum error allowed, if it is set to 0 the error wo't be checked till the end
    :return: parameters, error
    """
    w = initial_w
    for it in range(max_iters):
        w = w - (gamma() if isinstance(gamma, Scheduler) else gamma)/y.size * tx*(y-tx*w)
        if max_error > 0 and loss_function(np.dot(tx, w), y) < max_error:
            break
    return w, loss_function(y, tx*w)


def least_squares_sgd(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """
    Linear regression using stochastic gradient descent.

    :param y: label
    :param tx: input
    :param initial_w: initial weights
    :param max_iters: maximum iterators
    :param gamma: gradient factor, it can be a Scheduler
    :param batch_size: size of the batches
    :return: 
    """
    w = initial_w
    for it in range(max_iters):
        for yb, txb in batch_iter(y, tx, batch_size=batch_size):
            w = w - (gamma() if isinstance(gamma, Scheduler) else gamma)/yb.size * txb*(yb-txb*w)
    return w, mse(y, tx*w)


def least_squares_msgd(y, tx, initial_w, max_iters, gamma, momentum=0.1, batch_size=1):
    """
    Linear regression using stochastic gradient descent with momentum.

    :param y: label
    :param tx: input
    :param initial_w: initial weights
    :param max_iters: maximum iterators
    :param gamma: gradient factor, it can be a Scheduler
    :param momentum: momentum factor,it can be a Scheduler
    :param batch_size: size of the batches
    :return:
    """
    w = initial_w
    acc_momentum = 0
    for it in range(max_iters):
        for yb, txb in batch_iter(y, tx, batch_size=batch_size):
            acc_momentum = - (gamma() if isinstance(gamma, Scheduler) else gamma)/yb.size * txb*(yb-txb*w) + \
                           (momentum() if isinstance(momentum, Scheduler) else gamma) * acc_momentum
            w = w - acc_momentum
    return w, mse(y, tx*w)


def least_squares(y, tx):
    """"
    Least squares regression using normal equations.

    It assumes that tx*x is invertible.

    :param y: label
    :param tx: inputs
    :return: (Xt*X)^−1*Xt*y
    """
    return np.dot(np.dot(inv(np.dot(tx, np.transpose(tx, (1, 0)))), tx), y)
