import numpy as np
from src.functions.distance import Square
from src.preconditioning.normalization import MinMaxNormalizer


class LDA:
    """
    This has been heavily influenced by the following article:
    http://sebastianraschka.com/Articles/2014_python_lda.html

    This changes the axis of the features, decreasing the number of them,
    to maximize the separability of the classes.

    Maximizes := distance(mean_1-mean_2)/(variance_1 + variance_2)
    """
    def __init__(self, x: np.array, y: np.array, n_classes=2, out_features=3):
        y = y.flatten()
        n_features = x.shape[1]
        mean_vectors = []
        for i in range(n_classes):
            mean_vectors.append(np.mean(x[y == i], axis=0))

        s_w = np.zeros((n_features, n_features))
        for i, mv in zip(range(n_classes), mean_vectors):
            for sample in x[y == i]:
                sample, mv = sample.reshape(-1, 1), mv.reshape(n_features, 1)
                s_w += (sample-mv).dot((sample-mv).T)
        global_mean = np.mean(x, axis=0)

        s_b = np.zeros((n_features, n_features))
        for i, mean_vec in enumerate(mean_vectors):
            n = x[y == i].shape[0]
            mean_vec = mean_vec.reshape(-1, 1)
            global_mean = global_mean.reshape(-1, 1)
            s_b += n * (mean_vec - global_mean).dot((mean_vec - global_mean).T)

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        self.w = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(0, out_features)])

    def __call__(self, x):
        return np.dot(x, self.w).astype(float)

import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    idx = ~np.logical_or(np.any(data == -999, axis=1), np.any(data == 0, axis=1))
    data[:, 2:] = np.load(file=path + 'filled_dataset.npy')

    labels = data[:, 1]
    X = (data[:, 2:])
    X = MinMaxNormalizer()(X)
    X_lda = LDA(X, labels)(X)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    n_pts = int(1000)
    for label, marker, color in zip((0, 1), ('^', 'o'), ('blue', 'red')):
        r = int(np.random.uniform(0, X_lda[labels == label].shape[0]-n_pts-1))
        ax.scatter(X_lda[labels == label, 0][r:r+n_pts],
                   X_lda[labels == label, 1][r:r+n_pts],
                   X_lda[labels == label, 2][r:r+n_pts],
                   marker=marker)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    pass
