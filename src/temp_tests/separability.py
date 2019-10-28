from src.model.regression.linear_model import LinearModel
from src.optimization.linear import LinearSGD
from src.functions.loss import LogCosh
from src.utils.data_manipulation import split
from src.preconditioning.normalization import MinMaxNormalizer
import numpy as np
from src.preconditioning.feature_filling import MeanFilling
import os


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]
    data = np.load(file=path + '\\resources\\' + 'train.npy')

    data = MinMaxNormalizer()(MeanFilling(data[:, 2:])(data[:, 2:]))
    loss = LogCosh()
    model = LinearModel((data.shape[1], 1))
    kwargs = {'batch_size': 25, 'loss': loss, 'lr': 10**-1, 'epochs': 1000, 'epoch_step': (100, .75)}
    optimizer = LinearSGD()
    n_models = 2

    separability = []
    for i in range(2, data.shape[1]):
        mean1 = np.mean(data[data[:, 1] == 0, i])
        mean2 = np.mean(data[data[:, 1] == 1, i])
        var1 = np.var(data[data[:, 1] == 0, i])
        var2 = np.var(data[data[:, 1] == 1, i])
        sum_var = np.max([var1 + var2, 0.000001])
        print(sum_var)
        separability.append(np.abs(mean1 - mean2) / sum_var)
    print(separability)
