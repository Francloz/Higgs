import numpy as np
from src.functions.function import Derivable


class Loss:
    """
    Generic loss function template.

    :param x: output matrix of dim b x k
    :param y: label matrix of dim b x k
    """
    def __call__(self, x, y):
        pass

    def gradient(self, x, y):
        pass


class MSE(Loss):
    """
    Generic MSE loss
    """
    def __str__(self):
        return "MSE"

    def __call__(self, x, y):
        """
        Computes the loss of the regression.
        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: loss value
        """
        return np.mean((x - y)**2)

    def gradient(self, x, y):
        """
        Computes the gradient of the weight matrix.
        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: gradient matrix
        """
        return x - y


class MAE(Loss):
    def __str__(self):
        return "MAE"

    def __call__(self, x, y):
        """
        Computes the loss of the regression.
        :param x: input matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: loss value
        """
        return np.mean(np.abs(x - y))

    def gradient(self, x, y):
        """
        Computes the gradient of the weight matrix.
        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: gradient matrix
        """
        error = x-y
        error = np.where(error >= 0, 1, -1)
        return error/y.shape[0]


class LogCosh(Loss):
    def __str__(self):
        return "LogCosh"

    def __call__(self, x, y):
        """
        Computes the loss.
        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: gradient matrix
        """
        return np.sum(np.log(np.cosh(x-y)))

    def gradient(self, x, y):
        """
        Computes the gradient.
        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: gradient matrix
        """
        error = x - y
        error = np.sinh(error)/np.cosh(error)
        return error / y.shape[0]


class Quantile(Loss):
    def __str__(self):
        return "Quantile"

    def __init__(self, gamma=0.1):
        """
        Class constructor.
        :param gamma: factor associated with the error sign
        """
        self.gamma = gamma

    def __call__(self, x, y):
        """
        Computes the loss of the regression.

        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: loss value
        """
        error = x-y
        return np.sum(np.where(error < 0,
                               (self.gamma-1) * np.abs(error),
                               self.gamma * np.abs(error)))

    def gradient(self, x, y):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: gradient matrix
        """
        error = x - y
        error = np.where(error >= 0, self.gamma, -(1 - self.gamma))
        return error / y.shape[0]


class Huber(Loss):
    def __str__(self):
        return "Huber"

    def __init__(self, delta=.1):
        """
        Class constructor
        :param delta: parameter of the huber loss
        """

        self.delta = delta

    def __call__(self, x, y):
        """
        Computes the loss of the regression.

        :param x: output matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: loss value
        """
        return np.sum(np.where(np.abs(x - y) < self.delta,
                               0.5 * ((x - y) ** 2),
                               self.delta * np.abs(x - y) - 0.5 * (self.delta ** 2)))

    def gradient(self, x, y):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x k
        :param y: label matrix of dim b x k
        :return: gradient matrix
        """
        error = x - y
        aux_abs = np.where(error >= 0, 1, -1)

        error = np.where(np.abs(error) < self.delta,
                         error,
                         self.delta * aux_abs)
        return error / y.shape[0]

