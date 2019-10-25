import numpy as np


class Normalizer:
    def __call__(self, data, *args):
        """
        Normalizes the data.
        :param data:
        :return: Normalized data.
        """
        return


class GaussianNormalizer(Normalizer):
    def __str__(self):
        return "GaussianNormalizer"

    def __init__(self):
        self.means = None
        self.deviations = None

    def __call__(self, data, *args):
        """
        Normalizes the data using the Gaussian distribution.
        :param data: original data
        :return: normalized data
        """
        if not (self.means is None):
            diff = data - np.reshape(np.multiply(np.ones(data.shape),
                                                 np.reshape(self.means, (1, 0))), newshape=data.shape)

        else:
            self.means = np.mean(data, axis=0)
            diff = data - np.reshape(np.multiply(np.ones(data.shape),
                                                 np.reshape(self.means, (1, -1))[:, np.newaxis]), newshape=data.shape)
            self.deviations = np.expand_dims(np.sqrt(np.sum(diff**2, axis=0)/(diff.shape[0]-1)), axis=1)
        return diff/np.reshape(np.multiply(np.ones(data.shape),
                                           np.transpose(self.deviations, (1, 0))[:, np.newaxis]), newshape=data.shape)


class DecimalScaling(Normalizer):
    def __str__(self):
        return "DecimalScaling"

    def __init__(self):
        self.power = None

    def __call__(self, data, *args):
        """
        Normalizes the data using the decimal scaling normalization.
        :param data: original data
        :return: normalized data
        """
        if self.power is None:
            range = np.max(np.abs(data), axis=0)
            self.power = np.round(np.log10(range))
            self.power = np.expand_dims(np.power(10*np.ones(self.power.shape), self.power), axis=1)
        return data / np.reshape(np.multiply(np.ones(data.shape),
                                             np.transpose(self.power, (1, 0))[:, np.newaxis]), newshape=data.shape)


class MinMaxNormalizer(Normalizer):
    def __str__(self):
        return "MinMaxNormalizer"

    def __init__(self):
        self.min_max = None

    def __call__(self, data, *args):
        """
        Normalizes the data using the Min-Max normalization.
        :param data: original data
        :return: normalized data
        """
        if self.min_max is None:
            self.min_max = np.reshape(np.max(np.abs(data), axis=0), (1, -1))
        matrix = np.ones((data.shape[0], 1)).dot(self.min_max)
        return data / matrix


class GaussianOutlierRemoval(Normalizer):
    def __call__(self, data, **kwargs):
        """
        Removes outliers outside the desired probability quantile.
        :param data: original data
        :param max_prob: probability quantile of outliers (0.05, 0.01, 0.005 or 0.001). Defaults to 0.05.
        :return: data without elements with data inside the unlikely quantile
        """
        normal_data = GaussianNormalizer()(data)
        quantiles = {0.05: 1.96, 0.01: 2.58, 0.005: 2.83, 0.001: 3.25}

        closest = 0.05
        prob = kwargs['max_prob'] if 'max_prob' in kwargs else .05
        minimum = abs(prob - 0.05)

        for key in quantiles.keys():
            if abs(key - prob) < minimum:
                minimum = abs(key - prob)
                closest = key

        max_value = quantiles[closest]
        to_keep = np.abs(normal_data) < max_value  # or data < map[closest]
        to_keep = np.sum(to_keep, axis=1) == data.shape[1]

        data = data[to_keep]
        return data
