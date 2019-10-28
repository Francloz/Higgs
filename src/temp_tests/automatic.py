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
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]
    data = np.load(file=path + '\\resources\\' + 'train.npy')

    mask = [False, False, True, True, True, False, True, True, True, True, False, True, True, True,
            True, True, True, True, True, True, True, False, True, True, True, False,
            True, True, True, True, True, True]

    n_features = np.sum(mask)
    models = [LinearModel((n_features + 1, 1)),
              LinearModel((n_features + 1, 1)),
              Logistic(n_features + 1),
              LinearModel((n_features + 1, 1))]

    optimizers = [Ridge(),
                  LS(),
                  LogisticSGD(),
                  LinearSGD()]

    optimizer_kwargs = [[{'lambda_': i} for i in np.linspace(0.45, 0.55, 20)],
                        [{}],
                        [{'batch_size': 25, 'loss': LogCosh(), 'lr': 10**-1, 'epochs': 1000, 'regularize': r}
                         for r in np.logspace(0, 10, 20)],
                        [{'batch_size': 25, 'loss': LogCosh(), 'lr': 10**-1, 'epochs': 1000}]]

    normalizers = [
                   MinMaxNormalizer()
                   # , GaussianNormalizer()
                   # , DecimalScaling()
                  ]

    lrf = LinearRegressionFilling(data[:, mask], epochs=1)
    lrf.load(path + '/src/preconditioning/regression_filler_params.npy')

    filling_data = [
                    MeanFilling(data[:, mask]),
                    # MedianFilling(data[:, 2:]),
                    # ClassAverageFilling(data[:, 2:], data[:, 1], n_classes=2),
                    lrf
                    ]

    n_initial = 10
    np.random.seed(0)

    min_error = np.inf

    for model, optimizer, optimizer_kwargs in zip(models, optimizers, optimizer_kwargs):
        for filler in filling_data:
            for normalizer in normalizers:
                best_model_kwargs = None
                min_error_kwargs = np.inf
                for kwargs in optimizer_kwargs:
                    min_error = np.inf
                    best = None
                    for i in range(n_initial):
                        labels = np.reshape(data[:, 1], (-1, 1))
                        train, test = split(np.hstack([labels, np.hstack([np.reshape(np.ones(data.shape[0]), (-1, 1)),
                                                                          normalizer(filler(data[:, mask]))])]))

                        model.set_param(np.random.uniform(low=-1, high=1, size=(n_features + 1, 1)))
                        optimizer(model, train[:, 1:], np.reshape(train[:, 0], (-1, 1)), **kwargs)

                        prediction = np.where(model(test[:, 1:]) > .5, 1, 0)
                        correct_label = np.reshape(test[:, 0], (-1, 1))
                        error = np.sum(np.abs(prediction - correct_label))/prediction.size

                        if error < min_error:
                            min_error = error
                            best = model.get_params()
                    if min_error < min_error_kwargs:
                        min_error_kwargs = min_error
                        best_model_kwargs = best
                    print("Error for model", model, "using ", optimizer, dict([(a, str(x)) for a, x in kwargs.items()]),
                          " and ", normalizer, filler, " is ", min_error)
                np.save(arr=best_model_kwargs, file='./' + str(model) + str(optimizer) + str(filler) + str(normalizer)
                                                    + ("%0.5f" % min_error) + '.npy')
