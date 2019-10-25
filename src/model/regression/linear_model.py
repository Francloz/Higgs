from src.functions.loss import *
from src.model.model import Model


class LinearModel(Model):
    def __str__(self):
        return "LinearModel"

    def __init__(self, shape: tuple):
        """
        Class constructor.
        :param shape: shape of the weight matrix.
        """
        self.w = np.zeros(shape)

    def __call__(self, tx: np.array):
        """
        Evalueates the sample.
        :param tx: sample
        :return: predictions
        """
        return np.dot(tx, self.w)

    def set_param(self, w: np.array):
        """
        Sets a new the weight matrix.
        :param w: weight matrix
        """
        self.w = w

    def load(self, path: str):
        """
        Loads the weight matrix from the file.
        :param path: path to  the file
        """
        self.w = np.reshape(np.fromfile(path, sep=" ")[:self.w.size], self.w.shape)

    def save_model(self, path: str):
        """
        Saves the model to a file.
        :param path: path to the file
        """
        f = open(path, "w+")
        np.savetxt(f, self.w)
        f.close()

    def get_params(self):
        return self.w



