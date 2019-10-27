import os
import sys
sys.path.append(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0])

from src.optimization.linear import *
from src.model.classifier.logistic import Logistic
from src.functions.distance import L2
from src.utils.data_manipulation import *

from src.optimization.linear import *
from src.optimization.logistic import *
from src.preconditioning.normalization import *
from src.preconditioning.feature_filling import *
from src.functions.loss import LogCosh


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')

    models = [LinearModel((31, 1)),
              LinearModel((31, 1)),
              LinearModel((31, 1)),
              Logistic(31)]

    optimizers = [LinearSGD(),
                  Ridge(),
                  LS(),
                  LogisticSGD()]

    optimizer_kwargs = [[{'batch_size': 25, 'loss': LogCosh(), 'lr': 10**-1, 'epoch': 1000}],
                        [{'lambda_': i} for i in np.logspace(-5, 0, 15)],
                        [{}],
                        [{'batch_size': 25, 'loss': LogCosh(), 'lr': 10**-1, 'epoch': 1000, 'regularize': r}
                         for r in np.logspace(0, 10, 20)]]

    normalizers = [MinMaxNormalizer(),
                   # GaussianNormalizer(),
                   DecimalScaling()]

    filling_data = [MeanFilling(data[:, 2:]),
                    MedianFilling(data[:, 2:]),
                    ClassAverageFilling(data[:, 2:], data[:, 1], n_classes=2),
                    LinearRegressionFilling(data[:, 2:])]

    for model, optimizer, optimizer_kwargs in zip(models, optimizers, optimizer_kwargs):
        for filler in filling_data:
            for kwargs in optimizer_kwargs:
                for normalizer in normalizers:
                    labels = np.reshape(data[:, 1], (-1, 1))
                    train, test = split(np.hstack([labels, np.hstack([np.reshape(np.ones(data.shape[0]), (-1, 1)),
                                                                      normalizer(filler(data[:, 2:]))])]))
                    optimizer(model, train[:, 1:], np.reshape(train[:, 0], (-1, 1)), **kwargs)

                    prediction = np.where(model(test[:, 1:]) > .5, 1, 0)
                    correct_label = np.reshape(test[:, 0], (-1, 1))
                    print("Error for model", model, "using ", optimizer, kwargs, " and ", normalizer, filler, " is ", np.sum(np.abs(prediction - correct_label))/prediction.size)
