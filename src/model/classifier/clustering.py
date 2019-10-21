import numpy as np
from src.model.model import Model
from src.functions.distance import Distance, L1


class KNeighbourhoodCluster(Model):
    def __init__(self, x, y, distance : Distance, k):
        self.points = x
        self.labels = y
        self.distance = distance
        self.k = k

    def __call__(self, x):
        point_matrix = self.points.shape[0]
        point_matrix = np.repeat(x, point_matrix, axis=0)
        new_shape = (self.points.shape[0], x.size)
        point_matrix = np.reshape(point_matrix, new_shape)

        k_best = np.zeros((x.shape[0], self.k), dtype=int)

        for i in range(x.shape[0]):
            repeated_point = point_matrix[:, x.shape[1]*i:x.shape[1]*(i+1)]
            repeated_point = np.reshape(repeated_point, self.points.shape)
            distances = self.distance(repeated_point, self.points)
            idx = np.argpartition(a=distances, kth=self.k)
            best = self.labels[idx[:self.k]].flatten()
            k_best[i] = best

        predictions = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            votes = k_best[i]
            count_votes = np.bincount(votes)
            predictions[i] = count_votes.argmax()
        return predictions
