import numpy as np


def mse(x, y):
    return 1/x.size * np.sum((x - y)**2)


def mae(x, y):
    return 1/x.size * np.sum(np.abs(x-y))


def huber(x, y, delta):
    return np.sum(np.where(np.abs(x - y) < delta,
                  0.5 * ((x - y) ** 2),
                  delta * np.abs(x - y) - 0.5 * (delta ** 2)))


def log_cosh(x, y):
    return np.sum(np.log(np.cosh(x-y)))


def quantile(x, y, gamma):
    return np.sum(np.where(x < y,
                           (1-gamma) * abs(x - y),
                           gamma * abs(x - y)))
