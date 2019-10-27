import numpy as np
from src.model.regression.linear_model import LinearModel
from src.optimization.linear import Ridge
from src.preconditioning.normalization import *
from src.preconditioning.feature_filling import *
from src.visualization.plotting import plot
import os
"""
THis file will be deprecated soon.
"""


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    :param y:
    :param tx:
    :param lambda_:
    """


def ridge_regression(y, tx):
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]
    data = np.load(file=path + '\\resources\\' + 'train.npy')

    model = LinearModel((31, 1))
    optimizer = Ridge()
    normalizer = MinMaxNormalizer()
    filler = MeanFilling(tx)
    lambdas = np.array([i for i in np.linspace(0.42, 0.52, 1)])
    min_error = np.inf
    best = None
    n_iter = 15
    errors = []
    for lambda_ in lambdas:
        iter_best = None
        iter_error = np.inf
        for iter in range(n_iter):
            labels = np.reshape(y, (-1, 1))
            train, test = split(np.hstack([labels, np.hstack([np.reshape(np.ones(tx.shape[0]), (-1, 1)),
                                                              normalizer(filler(tx))])]), test_relative_size=.1)

            model.set_param(np.random.uniform(low=-1, high=1, size=(31, 1)))
            optimizer(model, train[:, 1:], np.reshape(train[:, 0], (-1, 1)), lambda_=lambda_)

            prediction = np.where(model(test[:, 1:]) > .5, 1, 0)
            correct_label = np.reshape(test[:, 0], (-1, 1))
            error = np.sum(np.abs(prediction - correct_label))/prediction.size

            if error < iter_error:
                iter_error = error
                iter_best = model.get_params()

        errors.append(iter_error)
        print(len(errors)/lambdas.size)

        if iter_error < min_error:
            min_error = iter_error
            best = iter_best

    # plot(lambdas, np.array(errors), label=('Lambda', 'Error', str(normalizer) + ' and ' + str(filler))).savefig(str(filler) + str(normalizer) + '.png')
    print("Error for model", model, "using ", normalizer, filler, " is ", min_error)

    return best



def lasso_regression(y, tx, lambda_):
    """
    Lasso regression using normal equations.

    :param y:
    :param tx:
    :param lambda_:
    """
    pass
