import numpy as np


class ActivationFunction:
    """
    Base class of an activation function.
    """
    def __call__(self, inputs):
        pass

    def derivative(self, x):
        pass


class Sigmoid(ActivationFunction):
    """
    Sigmoid function.
    R -> (-1,1)
    """
    def __call__(self, x):
        return (np.ones(x.shape) + np.exp(-x))**-1

    def derivative(self, x):
        return self(x)*(1-self(x))


class Identity(ActivationFunction):
    """
    Identity function.
    """
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1
