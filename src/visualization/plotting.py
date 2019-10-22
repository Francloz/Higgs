import numpy as np
import matplotlib.pyplot as plt

def plot(data: np.array, bounds: np.array):
    """
    Plots the data into a graph.
    :param data: array of points in a 2D or 3D space
    :param bounds: bounds of every dimension
    """
    pass


def plot_variances(x):
    """
    Plots the data into a graph.
    """
    means = np.mean(x, axis=0)
    variances = np.var(x, axis=0)
    max_distance = np.max(np.abs(x - means), axis=0)
    plt.figure()
    plt.errorbar(range(x.shape[1]), means, yerr=max_distance, color='b', linestyle='None', marker='o',
                 capsize=5, markersize=3, label='Variance')
    plt.errorbar(range(x.shape[1]), means, yerr=variances, color='r', linestyle='None', marker='o',
                 capsize=5, markersize=3, label='Value range')
    plt.xticks(range(x.shape[1]), range(x.shape[1]))
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.legend(loc='upper left')
    plt.gca().xaxis.grid(True)
    plt.show()


def plot_means(x, y, n_classes):
    """
    Plots the data into a graph.
    """
    y = y.flatten()
    plt.figure()
    for i, color in zip(range(n_classes), ('r', 'b', 'g')):
        means = np.mean(x[y == i], axis=0)
        variances = np.var(x[y == i], axis=0)
        plt.errorbar(range(x[y == i].shape[1]), means, yerr=variances, color=color, linestyle='None', marker='o',
                     capsize=5, markersize=3, label='Class '+str(i))
    plt.xticks(range(x.shape[1]), range(x.shape[1]))
    plt.xlabel('Features')
    plt.ylabel('Values')
    plt.legend(loc='upper left')
    plt.gca().xaxis.grid(True)
    plt.show()


def plot_correlations(x):
    """
    Plots the data into a graph.
    """

    correlations = np.zeros((x.shape[1], x.shape[1]))
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            correlations[i, j] = np.corrcoef(x[:, i], x[:, j])[0, 1]
    fig, ax = plt.subplots()
    plt.imshow(correlations, label='Correlation', interpolation='nearest')
    plt.xticks(range(x.shape[1]), range(x.shape[1]))
    plt.yticks(range(x.shape[1]), range(x.shape[1]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            text = ax.text(j, i, '%.1f' % (correlations[i, j]*10),
                           ha="center", va="center", color="w", fontsize=8)
    plt.legend(loc='upper left')
    plt.show()


def plot_feature(x, y, feature, n_classes=2):
    y = y.flatten()
    for label, marker, color in zip(range(n_classes), ('^', 'o', 's'), ('b', 'r', 'g')):
        plt.scatter(x[y == label, feature],
                    np.repeat(label, x[y == label].shape[0]))
    plt.xlabel('Values')
    plt.ylabel('Label')
    plt.show()


def plot_correlation(x, f1, f2):
    plt.scatter(x[:, f1],
                x[:, f2])
    plt.xlabel('Feature: ' + str(f1))
    plt.ylabel('Feature: ' + str(f2))
    plt.show()


from src.preconditioning.normalization import *
from src.utils.data_manipulation import *
import os

if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    train, test = split(data)
    train = GaussianOutlierRemoval()(train[1:])
    y = train[:, 1]
    tx = train[:, 2:]
    tx = MinMaxNormalizer()(tx)
    # plot_correlations(tx)
    # plot_variances(tx)
    # plot_means(tx, y, 2)
    # for i in range(tx.shape[1]):
    #     plot_feature(tx, y, i)
    plot_correlation(tx, 23, 24)
    pass
