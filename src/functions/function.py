import numpy as np


class Function:
    def __call__(self, x) -> np.array:
        pass


class Derivable:
    def deriv(self) -> Function:
        pass


class DerivableFunction(Function, Derivable):
    pass
