import numpy as np

"""
This file will be deprecated soon.
"""


def mse(x, y):
    """
    Returns the Mean Square Error between both vectors.

    This method assumes both are vectors with he same shape.

    :param x: first vector
    :param y: second vector
    :return: mse
    """
    return 1/x.size * np.sum((x - y)**2)


def mpe(x, y, p):
    """
    Returns the Mean P-power Error between both vectors.

    This method assumes both are vectors with he same shape.

    :param x: first vector
    :param y: second vector
    :return: mse
    """
    return 1/x.size * np.sum((x - y)**p)


def mae(x, y):
    """
    Returns the Mean Absolute Error between both vectors.

    This method assumes both are vectors with he same shape.

    :param x: first vector
    :param y: second vector
    :return: mse
    """
    return 1/x.size * np.sum(np.abs(x-y))


def huber(x, y, delta):
    """
    Returns the Mean Square Error between both vectors.
    It behaves as a MSE or MAE depending on the delta value, the bigger it is
    the higher the error has to be to use MAE.

    This method assumes both are vectors with he same shape.

    :param x: first vector
    :param y: second vector
    :param delta:
    :return: huber or smoothed mae error
    """
    return np.sum(np.where(np.abs(x - y) < delta,
                  0.5 * ((x - y) ** 2),
                  delta * np.abs(x - y) - 0.5 * (delta ** 2)))


def log_cosh(x, y):
    """
    Returns the Log_Cosh error between both vectors.
    It behaves as MAE with large errors and like MSE with small errors.

    This method assumes both are vectors with he same shape.

    :param x: first vector
    :param y: second vector
    :return: log-cosh error
    """
    return np.sum(np.log(np.cosh(x-y)))


def quantile(x, y, gamma):
    """
    Returns the Quantile Error between both vectors.
    It behaves like  MAE with weight associated with sign.

    This method assumes both are vectors with he same shape.

    :param x: first vector
    :param y: second vector
    :param gamma: weight of the positive errors
    :return: quantile error
    """
    return np.sum(np.where(x < y,
                           (1-gamma) * abs(x - y),
                           gamma * abs(x - y)))
