from utils.data_manipulation import *
from functions.loss import MSE


class Model:
    def __call__(self, tx: np.array):
        pass

    def set_param(self, w: np.array):
        pass

    def load_model(self, path: str):
        pass

    def save_model(self, path: str):
        pass