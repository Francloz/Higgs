import numpy as np
from src.functions.distance import Square
from src.preconditioning.normalization import MinMaxNormalizer


class FeatureExtraction:
    def __init__(self, x, **kwargs):
        pass

    def __call__(self, x, **kwargs):
        pass


class LDA:
    """
    This has been heavily influenced by the following article:
    http://sebastianraschka.com/Articles/2014_python_lda.html

    This changes the axis of the features, decreasing the number of them,
    to maximize the separability of the classes.

    Maximizes := distance(mean_1-mean_2)/(variance_1 + variance_2)
    """
    def __init__(self, x, y, n_features=3):
        idx = np.where(y == 0, True, False)

        mean_1 = np.mean(x[idx], axis=0)
        mean_2 = np.mean(x[~idx], axis=0)
        c1 = x[idx]
        c2 = x[~idx]
        s_w = np.zeros((x.shape[1], x.shape[1]))
        for i in range(c1.shape[0]):
            s_w += np.dot(np.reshape(c1[i]-mean_1, (-1, 1)), np.reshape(c1[i]-mean_1, (1, -1)))
        for i in range(c2.shape[0]):
            s_w += np.dot(np.reshape(c2[i]-mean_1, (-1, 1)), np.reshape(c2[i]-mean_1, (1, -1)))

        mean = np.mean(x, axis=0)
        s_b = np.dot(np.reshape(mean-mean_1, (-1, 1)), np.reshape(mean-mean_1, (1, -1)))
        s_b += np.dot(np.reshape(mean-mean_2, (-1, 1)), np.reshape(mean-mean_2, (1, -1)))

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(s_w).dot(s_b))
        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        print('Eigenvalues in decreasing order:\n')
        for i in eig_pairs:
            print(i[0])

        print('Variance explained:\n')
        eigv_sum = sum(eig_vals)
        for i,j in enumerate(eig_pairs):
            print('eigenvalue {0:}: {1:.10%}'.format(i+1, (j[0]/eigv_sum).real))

        self.w = np.reshape(eig_pairs[0][1], (-1, 1))
        i = 1
        while i < n_features:
            self.w = np.hstack((self.w, eig_pairs[i][1].reshape(-1, 1)))
            i += 1
        self.w = self.w.astype(float)

    def __call__(self, x):
        return np.dot(x, self.w)


class PCA:
    pass


def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y == cl], axis=0))
    return mean_vectors


def scatter_within(X, y):
    class_labels = np.unique(y)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_W = np.zeros((n_features, n_features))
    for cl, mv in zip(class_labels, mean_vectors):
        class_sc_mat = np.zeros((n_features, n_features))
        for row in X[y == cl]:
            row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
            class_sc_mat += (row-mv).dot((row-mv).T)
        S_W += class_sc_mat
    return S_W


def scatter_between(X, y):
    overall_mean = np.mean(X, axis=0)
    n_features = X.shape[1]
    mean_vectors = comp_mean_vectors(X, y)
    S_B = np.zeros((n_features, n_features))
    for i, mean_vec in enumerate(mean_vectors):
        n = X[y==i+1,:].shape[0]
        mean_vec = mean_vec.reshape(n_features, 1)
        overall_mean = overall_mean.reshape(n_features, 1)
        S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return S_B


def get_components(X, eig_vals, eig_vecs, n_comp):
    n_features = X.shape[1]
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.hstack([eig_pairs[i][1].reshape(-1, 1) for i in range(0, n_comp)])
    return W.astype(float)


import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')

    labels = data[:, 1]
    X = data[:, 2:]

    # Her's
    S_W, S_B = scatter_within(X, labels), scatter_between(X, labels)
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    W = get_components(X, eig_vals, eig_vecs, n_comp=3)
    print('EigVals: %s\n\nEigVecs: %s' % (eig_vals, eig_vecs))
    print('\nW: %s' % W)
    X_lda = X.dot(W)

    # Mine
    # X_lda = LDA(X, labels)(X)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label, marker, color in zip(
            (0, 1), ('^', 's', 'o'),('blue', 'red', 'green')):
        ax.scatter(X_lda[labels == label, 0][:1000],
                   X_lda[labels == label, 1][:1000],
                   X_lda[labels == label, 2][:1000],
                   marker=marker)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()