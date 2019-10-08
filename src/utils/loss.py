import numpy as np

"""
This file will be the combination of gradient.py and error.py 

Loss functions:
    - MAE
    - MSE
    - MPE
    - Huber
    - LogCosh
    - Quantile
"""


class Loss:
    def __call__(self, tx, y, w):
        pass

    def gradient(self, tx, y, w):
        pass


class MSE(Loss):
    """
    Multipurpose MSE loss function.

    W = matrix dim n x k
    tx = matrix of dim b x n
    y = matrix of dim b x k

    This class computes the mean squared error of the 'tx*W - y' rows and the gradient.
    """
    def __call__(self, tx, y, w):
        return np.mean((np.dot(tx, w) - y)**2, axis=1)

    def gradient(self, x, y, w):
        # Each column of w has gradient
        # 1/b t(txi) * (txi*Wj - yi)
        mean = np.zeros(w.shape)
        for i in range(x.shape[0]):
            column = np.expand_dims(x[i], axis=1)
            expanded_matrix = np.reshape(np.repeat(np.expand_dims(x[i], axis=1), w.shape[1]), w.shape)
            error = np.dot(x[i], w) - y[i]
            mean += np.dot(np.dot(x[i], w) - y[i], np.reshape(np.repeat(np.expand_dims(x[i], axis=1), w.shape[1]), w.shape))
        return mean/x.shape[0]

if __name__ == "__main__":
    w = np.zeros((2, 2))
    x = np.array([[1, 2],
                  [2, 3]])
    y = np.array([[1, 2],
                  [2, 3]])

    for i in range(100):
        w -= MSE().gradient(x, y, w)
    print(np.dot(x, w))
    pass
"""
delta = np.dot(self.act_fun(self.last_output) - param, self.act_fun.derivative(self.last_output))
err = np.dot(delta, np.transpose(self.last_input, (1, 0)))
self.weights -= (lr() if isinstance(lr, Scheduler) else lr)*err
"""


class MAE(Loss):
    def __call__(self, tx, y, w):
        return 1 / y.size * np.sum(np.abs(np.dot(tx, w) - y))

    def gradient(self, tx, y, w):
        pass  # return 1 / y.size * np.sum(tx.dot(y - np.dot(tx, w)))


class MPE(Loss):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, tx, y, w):
        return 1 / y.size * np.sum((np.dot(tx, w) - y) ** self.p)

    def gradient(self, tx, y, w):
        pass  # return 1/y.size*np.sum(tx.dot(y - np.dot(tx, w)), tx)


class Huber(Loss):
    def __init__(self, delta):
        self.delta = delta

    def __call__(self, tx, y, w):
        x = np.dot(tx, w)
        return np.sum(np.where(np.abs(x - y) < self.delta,
                               0.5 * ((x - y) ** 2),
                               self.delta * np.abs(x - y) - 0.5 * (self.delta ** 2)))

    def gradient(self, tx, y, w):
        pass


class LogCosh(Loss):
    def __call__(self, tx, y, w):
        return np.sum(np.log(np.cosh(np.dot(tx, w)-y)))

    def gradient(self, tx, y, w):
        pass


class Quantile(Loss):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, tx, y, w):
        x = np.dot(tx, w)
        return np.sum(np.where(x < y,
                               (1-self.gamma) * np.abs(x - y),
                               self.gamma * np.abs(x - y)))

    def gradient(self, tx, y, w):
        pass
