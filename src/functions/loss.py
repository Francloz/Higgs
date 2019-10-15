import numpy as np


class Loss:
    """
    Generic loss function template.

    W = matrix dim n x k
    x = matrix of dim b x n
    y = matrix of dim b x k
    """
    def __call__(self, x, y, w):
        pass

    def gradient(self, x, y, w):
        pass


class MSE(Loss):
    """
    Generic MSE loss
    """
    def __call__(self, x, y, w):
        """
        Computes the loss of the regression.

        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: loss value
        """
        return np.mean((np.dot(x, w) - y)**2)

    def gradient(self, x, y, w):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: gradient matrix
        """
        # for c in range(w.shape[1]):
        #    loss_v = np.expand_dims(np.dot(x, w[:, c])-y[:, c], axis=0)
        #    der[:, c] = np.dot(loss_v, x)

        error = np.dot(x, w) - y
        aux_der = np.dot(np.transpose(x, (1, 0)), error)
        return aux_der


class MAE(Loss):
    def __call__(self, x, y, w):
        """
        Computes the loss of the regression.

        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: loss value
        """
        return np.mean(np.abs(np.dot(x, w) - y))

    def gradient(self, x, y, w):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: gradient matrix
        """
        error = np.dot(x, w)-y
        error = np.where(error >= 0, 1, -1)
        derivative = np.dot(np.transpose(x, (1, 0)), error)
        return derivative/w.size


class LogCosh(Loss):
    def __call__(self, x, y, w):
        """
        Computes the loss of the regression.

        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: loss value
        """
        return np.sum(np.log(np.cosh(np.dot(x, w)-y)))

    def gradient(self, x, y, w):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: gradient matrix
        """
        error = np.dot(x, w) - y
        error = np.sinh(error)/np.cosh(error)
        aux_der = np.dot(np.transpose(x, (1, 0)), error)
        return aux_der / w.size


class Quantile(Loss):
    def __init__(self, gamma=0.1):
        """
        Class constructor.
        :param gamma: factor associated with the error sign
        """
        self.gamma = gamma

    def __call__(self, x, y, w):
        """
        Computes the loss of the regression.

        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: loss value
        """
        error = np.dot(x, w)-y
        return np.sum(np.where(error < 0,
                               (self.gamma-1) * np.abs(error),
                               self.gamma * np.abs(error)))

    def gradient(self, x, y, w):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: gradient matrix
        """
        error = np.dot(x, w) - y
        error = np.where(error >= 0, self.gamma, -(1 - self.gamma))
        derivative = np.dot(np.transpose(x, (1, 0)), error)
        return derivative / w.size


class Huber(Loss):
    def __init__(self, delta=.1):
        """
        Class constructor
        :param delta: parameter of the huber loss
        """

        self.delta = delta

    def __call__(self, x, y, w):
        """
        Computes the loss of the regression.

        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: loss value
        """
        output = np.dot(x, w)
        return np.sum(np.where(np.abs(output - y) < self.delta,
                               0.5 * ((output - y) ** 2),
                               self.delta * np.abs(output - y) - 0.5 * (self.delta ** 2)))

    def gradient(self, x, y, w):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: gradient matrix
        """
        error = np.dot(x, w) - y
        aux_abs = np.where(error >= 0, 1, -1)

        error = np.where(np.abs(error) < self.delta,
                         error,
                         self.delta * aux_abs)

        derivative = np.dot(np.transpose(x, (1, 0)), error)
        return derivative / w.size


class MSE(Loss):
    """
    Generic MSE loss
    """
    def __call__(self, x, y, w):
        """
        Computes the loss of the regression.

        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: loss value
        """
        return np.mean((np.dot(x, w) - y)**2)

    def gradient(self, x, y, w):
        """
        Computes the gradient of the weight matrix.
        :param x: input matrix of dim b x n
        :param y: label matrix of dim b x k
        :param w: weight matrix dim n x k
        :return: gradient matrix
        """
        # for c in range(w.shape[1]):
        #    loss_v = np.expand_dims(np.dot(x, w[:, c])-y[:, c], axis=0)
        #    der[:, c] = np.dot(loss_v, x)

        error = np.dot(x, w) - y
        aux_der = np.dot(np.transpose(x, (1, 0)), error)
        return aux_der