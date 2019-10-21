import numpy as np


class SpacePartition:
    def __init__(self, max_pts=100, axis=0):
        self.left = None
        self.right = None
        self.points = None
        self.max_pts = max_pts
        self.axis = axis
        self.median = 0

    def _partition(self):
        self.median = np.median(self.points, axis=1)[self.axis]
        self.left = SpacePartition(self.max_pts, self.axis + 1 if self.axis < self.points.shape[1] - 1 else 0)
        self.right = SpacePartition(self.max_pts, self.axis + 1 if self.axis < self.points.shape[1] - 1 else 0)

        for i in self.points.shape[0]:
            if self.points[i][self.axis] < self.median:
                self.left._insert(self.points[i])
            else:
                self.right._insert(self.points[i])

    # noinspection PyProtectedMember
    def _insert(self, x):
        if self.points.shape[0] == self.max_pts:
            self._partition()

        if self.left is None:
            if self.points is None:
                self.points = np.array([x])
            else:
                self.points = np.vstack([self.points, x])
        else:
            if x[self.axis] < self.median:
                self.left._insert(x)
            else:
                self.right._insert(x)

    def insert(self, x: np.array, label: int):
        for r in range(x.shape[0]):
            self._insert(np.append(x[r, :], label))

    def get_k_closest(self, k, x):
        if self.left is None:
            distances = np.sum((self.points -
                                np.reshape(np.repeat(np.append(x, 0), self.points.shape[0]),
                                           self.points.shape)[:, :-1])**2,
                               axis=0)
        pass

    def get_closest(self, k, x):
        pass
