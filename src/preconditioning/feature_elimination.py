import numpy as np
from src.preconditioning.normalization import MinMaxNormalizer


class VarianceThreshold:
    def __init__(self, x, **kwargs):
        x = MinMaxNormalizer()(x)
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs else 0.01
        self.means = np.expand_dims(np.sum(x, axis=0)/x.shape[0], axis=1)
        diff = x - np.reshape(np.multiply(np.ones(x.shape),
                                          np.transpose(self.means, (1, 0))[:, np.newaxis]), newshape=x.shape)
        self.deviations = np.expand_dims(np.sqrt(np.sum(diff**2, axis=0)/(diff.shape[0]-1)), axis=1)
        self.idx = np.where(self.deviations < self.threshold, False, True)
        pass

    def __call__(self, x):
        return x[:, self.idx.flatten()]


class CorrelationThreshold:
    def __init__(self, x, **kwargs):
        x = MinMaxNormalizer()(x)
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs else 0.95
        self.correlations = np.zeros((x.shape[1], x.shape[1]))
        self.idx = np.ones(x.shape[1])

        disjoint = np.array(range(0, x.shape[1]))
        for i in range(x.shape[1]):
            for j in range(i+1, x.shape[1]):
                self.correlations[i, j] = np.corrcoef(x[:, i], x[:, j])[0, 1]
                if self.correlations[i, j] > self.threshold:
                    type_j = disjoint[j]
                    while type_j != disjoint[type_j]:
                        type_j = disjoint[type_j]
                    type_i = disjoint[i]
                    while type_i != disjoint[type_i]:
                        type_i = disjoint[type_i]
                    disjoint[type_i] = type_j
        for i in range(x.shape[1]):
            self.idx[i] = True if disjoint[i] == i else False
        self.idx = self.idx.astype(dtype=bool)

    def __call__(self, x):
        return x[:, self.idx.flatten()]


def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += jump

import os

if __name__ == '__main__':
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')[:, 2:]
    for th in drange(0.01, 1, 0.01):
        dr = VarianceThreshold(data, threshold=th)
        aux = dr(data)
        print((int(sum(~dr.idx)), dr.threshold))
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')[:, 2:]
    for th in drange(.99, 1, 0.001):
        dr = CorrelationThreshold(data, threshold=th)
        aux = dr(data)
        print((int(sum(~dr.idx)), dr.threshold))
