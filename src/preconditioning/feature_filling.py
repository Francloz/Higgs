import numpy as np
from src.model.regression.linear_model import LinearModel
from src.optimization.linear import *
from src.preconditioning.normalization import MinMaxNormalizer

# See https://dev.acquia.com/blog/how-to-handle-missing-data-in-machine-learning-5-techniques/09/07/2018/19651


class MeanFilling:
    def __str__(self):
        return "MeanFilling"

    def __init__(self, x):
        self.means = np.array([np.mean(x[np.logical_and(x[:, i] != -999, x[:, i] != 0), i], axis=0) for i in range(x.shape[1])])

    def __call__(self, x):
        x_aux = np.array(x, copy=True)
        for i in range(x_aux.shape[1]):
            x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i] = \
                np.repeat(self.means[i], x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i].shape[0])
        return x_aux


class MedianFilling:
    def __str__(self):
        return "MedianFilling"

    def __init__(self, x):
        self.medians = np.array([np.median(x[np.logical_and(x[:, i] != -999, x[:, i] != 0), i], axis=0) for i in range(x.shape[1])])

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
        self.means = np.sum(np.array([np.sum(y == i) *
                                      np.mean(x[np.logical_and(np.logical_and(x[:, i] != -999,
                                                               x[:, i] != 0), (y == i))],
                                              axis=0)/y.shape[0] for i in range(n_classes)]), axis=0)

    def __call__(self, x):
        x_aux = np.array(x, copy=True)
        for i in range(x_aux.shape[1]):
            x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i] = \
                np.repeat(self.means[i], x_aux[np.logical_or(x_aux[:, i] == -999, x_aux[:, i] == 0), i].shape[0])
        return x_aux


class LinearRegressionFilling:
    def __str__(self):
        return "LinearRegressionFilling"

    def __init__(self, x, epochs=0):
        x = x[~np.logical_or(np.any(x == -999, axis=1), np.any(x == 0, axis=1))]
        self.normalizer = MinMaxNormalizer()
        x = self.normalizer(x)
        self.models = [LinearModel((x.shape[1]-1, 1)) for i in range(x.shape[1])]
        optimizer = LinearSGD()
        mask = np.repeat(True, x.shape[1])
        for i in range(x.shape[1]):
            mask[i] = False
            step_epoch = 50
            lr_reduc = 0.9
            lr = 10**-1
            for epoch in range(int(epochs/step_epoch)):
                optimizer(self.models[i],  x[:, mask], np.reshape(x[:, i], (-1, 1)),
                          lr=lr, epochs=step_epoch, batch_size=20, num_batches=100)
                lr *= lr_reduc
            sys.stdout.write("\rLearned %d\r" % i)
            mask[i] = True

    def save(self, file):
        np.save(arr=np.concatenate([model.get_params().flatten() for model in self.models]), file=file)

    def load(self, file):
        all_params = np.load(file)
        shape = self.models[0].get_params().shape
        size = self.models[0].get_params().size
        for i in range(len(self.models)):
            self.models[i].set_param(np.reshape(all_params[i*size:(i+1)*size], shape))

    def __call__(self, x):
        mask = np.repeat(True, x.shape[1])
        indexes = [np.logical_or(x[:, i] == -999, x[:, i] == 0) for i in range(x.shape[1])]
        aux = self.normalizer(MedianFilling(x)(x))
        for i in range(x.shape[1]):
            mask[i] = False
            masked_features = aux[:, mask]
            masked_features = masked_features[indexes[i]]
            output = self.models[i](masked_features)
            aux[indexes[i], i] = output.flatten()
            mask[i] = True
        return aux



import os

if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    x = data[:, 2:]
    y = data[:, 1]
    # x2 = MeanFilling(x)(x)
    # x1 = ClassAverageFilling(x, y, 2)(x)
    # x3 = MedianFilling(x)(x)
    x, _ = split(x, .5)
    filler = LinearRegressionFilling(x, epochs=1000)
    filler.save("./regression_filler_params.npy")
    filler.load("./regression_filler_params.npy")
    x4 = filler(x)
    np.save(arr=x4, file=path + 'filled_dataset.npy')
    pass
