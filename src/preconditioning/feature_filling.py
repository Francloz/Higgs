import numpy as np

# See https://dev.acquia.com/blog/how-to-handle-missing-data-in-machine-learning-5-techniques/09/07/2018/19651


class MeanFilling:
    def __init__(self, x):

        for i in range(x.shape[1]):
            feature = x[np.logical_and(x[:, i] != -999, x[:, i] != 0), i]
            pass
        self.means = np.array([np.mean(x[np.logical_and(x[:, i] != -999, x[:, i] != 0), i], axis=0) for i in range(x.shape[1])])
        pass

    def __call__(self, x):
        x_aux = np.array(x, copy=True)
        for i in range(x_aux.shape[1]):
            x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i] = \
                np.repeat(self.means[i], x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i].shape[0])
        return x_aux


class MedianFilling:
    def __init__(self, x):
        self.medians = np.array([np.median(x[np.logical_and(x[:, i] != -999, x[:, i] != 0), i], axis=0) for i in range(x.shape[1])])
        pass

    def __call__(self, x):
        x_aux = np.array(x, copy=True)
        for i in range(x_aux.shape[1]):
            x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i] = \
                np.repeat(self.medians[i], x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i].shape[0])
        return x_aux


class ClassAverageFilling:
    def __init__(self, x, y, n_classes):
        self.n_classes = n_classes
        # Smth wrong here
        self.means = np.sum(np.array([np.sum(y == i)*np.mean(x[np.logical_and(np.logical_and(x[:, i] != -999,
                                                                                             x[:, i] != 0), (y == i))],
                                                             axis=0)/y.shape[0] for i in range(n_classes)]), axis=0)
        pass

    def __call__(self, x):
        x_aux = np.array(x, copy=True)
        for i in range(x_aux.shape[1]):
            x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i] = \
                np.repeat(self.means[i], x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i].shape[0])
        return x_aux


import os
if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    x = data[:, 2:]
    y = data[:, 1]
    x2 = MeanFilling(x)(x)
    x1 = ClassAverageFilling(x,y,2)(x)
    x3 = MedianFilling(x)(x)
    pass
