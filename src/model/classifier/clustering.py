import numpy as np
from model.model import Model


class KNeighbourhoodCluster(Model):
    def __init__(self, tx, y, distance, k):
        self.points = tx
        self.labels = y
        self.distance = distance
        self.k = k

    def __call__(self, x):
        k_best = np.array([])
        for i in range(self.points.shape[0]):
            d = self.distance(x, self.points[i])
            np.append(k_best, [self.labels[i], d])
            if k_best.shape[0] > self.k:
                worst = 0
                for k_it in range(self.k):
                    if k_best[worst][1] < k_best[k_it][1]:
                        worst = k_it
                np.delete(k_best, worst, axis=0)
        votes = np.zeros(self.k)
        for i in np.nditer(k_best[:, 1]):
            votes[i] += 1

        best = (0, votes[0])
        for i in range(1, votes.size):
            if best[1] < votes[i]:
                best = (i, votes[i])
        return best[0]



