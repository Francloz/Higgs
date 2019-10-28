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

def plot_hist(x, y, n_classes=2):
    # remove -999 and 0 data
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i,j] == -999:
                x[i,j] = None
            if x[i,j] == 0 and j != 22: # 0 seems to be a needed value for feature 22
                x[i,j] = None

    label_check = 0
    for i in range(x.shape[1]):     # 30 features
        for label in zip(range(n_classes)): # 2 sets labels
            # for j in range(x[y==label].shape[0]): # rows of data entries
            #     if x[y==label][j,i] != -999:    
            #         if label_check == 1:
            #             plt.hist(x[y==label][j,i], bins='auto', alpha = 0.5, label = 'Higgs')
            #         else:
            #             plt.hist(x[y==label][j,i], bins='auto', alpha = 0.5, label = 'N/A')
            if label_check == 1:
                plt.hist(x[y==label][:,i], bins='auto', alpha = 0.5, label = 'Higgs')
            else:
                plt.hist(x[y==label][:,i], bins='auto', alpha = 0.5, label = 'N/A')
            label_check += 1
        label_check = 0
        plt.legend(loc='upper right')
        plt.title("Histogram with feature " + str(i))
        plt.xlabel('Feature')
        plt.ylabel('Count')
        plt.savefig('Feature' + str(i) + '.png', bbox_inches='tight')
        plt.show()


def get_feature_percents(x, y, n_classes=2):
    y = y.flatten()
    size = range(x.shape[1])
    for i in range(x.shape[1]):
        sum = np.sum(x[:,i]==-999)
        percent = sum/175000
        print(sum, sum/175000)
        if percent > .70:
            x[:,i] = 0
    print("break")

def plot_correlation(x, f1, f2):
    plt.scatter(x[:, f1],
                x[:, f2])
    plt.xlabel('Feature: ' + str(f1))
    plt.ylabel('Feature: ' + str(f2))
    plt.show()


from src.preconditioning.normalization import MinMaxNormalizer
from src.preconditioning.feature_filling import MeanFilling
from src.utils.data_manipulation import *
import os

if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')

    data[:, 2:] = MinMaxNormalizer()(MeanFilling(data[:, 2:])(data[:, 2:]))

    y = data[:, 1]
    tx = data[:, 2:]

    # plot_correlations(tx)
    # plot_variances(tx)
    # plot_means(tx, y, 2)
    # for i in range(tx.shape[1]):
    #    plot_feature(tx, y, i)

    metric = np.array([0.036434259483545436, 0.01655890915248768, 1.3150692309097907, 0.007325621143385119, 1.0368769447467558, 1.0499657756178657, 0.055327876381386885, 0.008071661275922438, 1.336160314444832, 3.047905391034283, 0.3075455658833619, 0.021964506938275646, 0.196633025986905, 0.3490715850335407, 0.04397869479743367, 0.069910177706796, 0.05036753873395129, 0.018576495119375442, 0.04833707112142514, 0.01243260601796758, 1.4634146341463414, 0.011154814882818392, 0.06629926398224029, 0.49738987958897907, 0.37120208255219167, 0.12524476197090162, 0.2688579029295658, 0.014512364099364499])
    y = np.where(metric > 10000, 10000, metric)
    x = np.array(range(1, 30-1))
    fig, ax = plt.subplots()
    ax.scatter(x, y, color='r', marker='^')

    for i in range(x.size):
        ax.annotate(i+1, (x[i] - .4, y[i] ))
    plt.show()

    # get_feature_percent(tx, y)
    # plot_hist(tx, y)
    # plot_correlation(tx, 23, 24)

    pass

