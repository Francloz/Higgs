import numpy as np


class Function:
    def __call__(self, x) -> np.array:
        pass


class Derivable(Function):
    def gradient(self, x) -> np.array:
        pass
