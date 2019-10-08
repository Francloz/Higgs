import numpy as np

"""
This file will be the combination of gradient.py and error.py 

Loss functions:
    - MAE
    - MSE
    - Huber
    - LogCosh
    - Quantile
"""


class Loss:
    def __call__(self, x, y, w):
        pass

    def gradient(self, x, y, w):
        pass


class MSE(Loss):
    """
    Multipurpose MSE loss function.

    W = matrix dim n x k
    tx = matrix of dim b x n
    y = matrix of dim b x k

    This class computes the mean squared error of the 'tx*W - y' rows and the gradient.
    """
    def __call__(self, x, y, w):
        return np.mean((np.dot(x, w) - y)**2)

    def gradient(self, x, y, w):
        der = np.zeros(w.shape)
        # m = np.transpose((np.dot(x, w) - y), (1, 0))
        for c in range(w.shape[1]):
            loss_v = np.expand_dims(np.dot(x, w[:, c])-y[:, c], axis=0)
            der[:, c] = np.dot(loss_v, x)
        # aux_der = np.dot(m, x)
        # print(np.sum(der-aux_der)*100000)
        return der,


class MAE(Loss):
    def __call__(self, x, y, w):
        return np.mean(np.abs(np.dot(x, w) - y))

    def gradient(self, x, y, w):
        der = np.zeros(w.shape)
        for c in range(w.shape[1]):
            loss_v = np.expand_dims(np.dot(x, w[:, c])-y[:, c], axis=0)
            loss_v = np.where(loss_v >= 0, 1, -1)
            der[:, c] = np.dot(loss_v, x)
        return der/w.size


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


if __name__ == "__main__":
    w1 = np.zeros((2, 2), dtype=np.double)
    x = np.array([[1, 2],
                  [2, 3],
                  [0, 0],
                  [-1, -1]], dtype=np.double)
    y = np.array([[1, 2],
                  [2, 3],
                  [0, 0],
                  [-1, -1]], dtype=np.double)
    loss = MAE()
    for i in range(1, 1000000):
        g1 = loss.gradient(x, y, w1)

        if i % 1000 == 0:
            print(loss(x, y, w1))

        w1 -= 10**(-5)*g1

    print(w1)
