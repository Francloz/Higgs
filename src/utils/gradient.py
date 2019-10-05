import numpy as np


def linear_gradient_mse(y, tx, w):
    """
    Computes the gradient of the MSE error.
    The target and input must match the valid sample conditions.

    :param y: target
    :param tx: input
    :param w: parameters of the linear regression
    """
    return -1/y.size*np.sum(tx.dot(y - tx*w))


def linear_gradient_mpe(y, tx, w, p):
    """
    Computes the gradient of the Mean P-power Error.
    The target and input must match the valid sample conditions.

    :param y: target
    :param tx: input
    :param w: parameters of the linear regression
    :param p: power (must not be 1)
    """
    # p(yi - xiw)^(1-p)
    return -1/y.size*np.sum(tx.dot(y - tx*w)**(p-1))


def linear_huber_loss(y, tx, w):
    """
    Computes the mean error minimization direction of the MAE error.
    The target and input must match the valid sample conditions.

    :param y: target
    :param tx: input
    :param w: parameters of the linear regression
    """
    return -1/y.size*np.sum(w - y / tx)
