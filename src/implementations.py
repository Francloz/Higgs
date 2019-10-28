import os
import sys
sys.path.append(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0])

from src.model.regression.linear_model import LinearModel
from src.model.classifier.logistic import Logistic
from src.optimization.logistic import LogisticSGD
from src.optimization.linear import *
from src.preconditioning.feature_filling import *
from src.preconditioning.normalization import *
from src.utils.data_manipulation import *
from src.functions.loss import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    np.random.seed(0)
    data = np.hstack([np.reshape(y, (-1, 1)), tx])

    model = LinearModel((tx.shape[1], y.shape[1]))
    optimizer = LinearGD()

    training, test = split(data)
    model.set_param(initial_w)
    optimizer(model, training[:, y.shape[1]:], training[:, :y.shape[1]],
              epochs=max_iters, epoch_step=(max_iters, 1),
              num_batches=1, lr=gamma, regularize=0,
              loss=MSE())
    error = MSE()(model(test[:, y.shape[1]:]), test[:, :y.shape[1]])

    return model.get_params(), error


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    np.random.seed(0)
    data = np.hstack([np.reshape(y, (-1, 1)), tx])

    model = LinearModel((tx.shape[1], y.shape[1]))
    optimizer = LinearSGD()

    training, test = split(data)
    model.set_param(initial_w)
    optimizer(model, training[:, y.shape[1]:], training[:, :y.shape[1]],
              epochs=max_iters, epoch_step=(max_iters, 1),
              num_batches=1, batch_size=1, lr=gamma, regularize=0,
              loss=MSE())
    error = MSE()(model(test[:, y.shape[1]:]), test[:, :y.shape[1]])

    return model.get_params(), error


def least_squares(y, tx):
    np.random.seed(0)
    data = np.hstack([np.reshape(y, (-1, 1)), tx])

    model = LinearModel((tx.shape[1], y.shape[1]))
    optimizer = LS()

    training, test = split(data)
    optimizer(model, training[:, y.shape[1]:], training[:, :y.shape[1]])
    error = MSE()(model(test[:, y.shape[1]:]), test[:, :y.shape[1]])

    return model.get_params(), error


def ridge_regression(y, tx, lambda_):
    np.random.seed(0)
    data = np.hstack([np.reshape(y, (-1, 1)), tx])

    model = LinearModel((tx.shape[1], y.shape[1]))
    optimizer = Ridge()

    training, test = split(data)

    optimizer(model, training[:, y.shape[1]:], training[:, :y.shape[1]], lambda_=lambda_)
    error = MSE()(model(test[:, y.shape[1]:]), test[:, :y.shape[1]])

    return model.get_params(), error


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    np.random.seed(0)
    data = np.hstack([np.reshape(y, (-1, 1)), tx])

    model = LinearModel((tx.shape[1], y.shape[1]))
    optimizer = LS()

    training, test = split(data)
    model.set_param(initial_w)
    optimizer(model,  training[:, y.shape[1]:], training[:, :y.shape[1]],
              epochs=max_iters, epoch_step=(max_iters, 1),
              num_batches=1, batch_size=1, lr=gamma, regularize=0)
    error = MSE()(model(test[:, y.shape[1]:]), test[:, :y.shape[1]])

    return model.get_params(), error


def re_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    np.random.seed(0)
    data = np.hstack([np.reshape(y, (-1, 1)), tx])

    model = LinearModel((tx.shape[1], y.shape[1]))
    optimizer = LS()

    training, test = split(data)
    model.set_param(initial_w)
    optimizer(model, training[:, y.shape[1]:], training[:, :y.shape[1]],
              epochs=max_iters, epoch_step=(max_iters, 1),
              num_batches=1, batch_size=1, lr=gamma, regularize=lambda_)
    error = MSE()(model(test[:, y.shape[1]:]), test[:, :y.shape[1]])

    return model.get_params(), error


"""
# Example run:

if __name__ == "__main__":
    tx = np.array([[0, 0],
                   [1, 4],
                   [2, 7],
                   [3, 8],
                   [4, 8]])
    y = np.array([[0,
                   1,
                   1,
                   1,
                   1]]).T

    _, error = ridge_regression(y, tx, 1)
    print(error)
    _, error = least_squares(y, tx)
    print(error)
    _, error = least_squares_GD(y, tx, np.zeros((tx.shape[1], y.shape[1])), 1000, 0.01)
    print(error)
    _, error = least_squares_SGD(y, tx, np.zeros((tx.shape[1], y.shape[1])), 1000, 0.01)
    print(error)
    _, error = logistic_regression(y, tx, np.zeros((tx.shape[1], y.shape[1])), 1000, 0.01)
    print(error)
    _, error = re_logistic_regression(y, tx, 2, np.zeros((tx.shape[1], y.shape[1])), 1000, 0.01)
    print(error)
 """
