import numpy as np
from model.model import Model


class KNeighbourhoodCluster(Model):
    def __init__(self, tx, y, distance, k):
        self.points = tx
        self.labels = y
        self.distance = distance
        self.k = k

    def __call__(self, x):
        k_best = np.array([[float(self.distance(x, self.points[0])), float(self.labels[0])]])
        for i in range(1, self.points.shape[0]):
            d = self.distance(x, self.points[i])
            k_best = np.vstack([k_best, np.array([float(d), float(self.labels[i])])])
            if len(k_best) > self.k:
                worst = 0
                for k_it in range(self.k):
                    if k_best[worst][0] < k_best[k_it][0]:
                        worst = k_it
                k_best = np.delete(k_best, k_best[worst][1], axis=0)
        votes = np.zeros(self.k)
        for i in np.nditer(k_best[:, 1]):
            votes[int(i)] += 1

        best = (0, votes[0])
        for i in range(1, votes.size):
            if best[1] < votes[i]:
                best = (i, votes[i])
        return np.array(best[0])



