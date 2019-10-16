import numpy as np


class Distance:
    def __call__(self, x: np.array, y: np.array):
        pass


class Square(Distance):
    def __call__(self, x: np.array, y: np.array):
        """
        Computes the square distance between two points.
        :param x: first point
        :param y: second point
        :return: absolute distance between the points
        """
        np.sum((x - np.reshape(y, x.shape))**2)


class L2(Distance):
    def __call__(self, x: np.array, y: np.array):
        """
        Computes the L2 distance between two points.
        :param x: first point
        :param y: second point
        :return: absolute distance between the points
        """
        np.sqrt(np.sum((x - np.reshape(y, x.shape))**2))


class L1(Distance):
    def __call__(self, x: np.array, y: np.array):
        """
        Computes the absolute distance between two points.
        :param x: first point
        :param y: second point
        :return: absolute distance between the points
        """
        np.sqrt(np.sum(np.abs(x - np.reshape(y, x.shape))))
