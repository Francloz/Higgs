import numpy as np
from src.functions.function import Derivable


class ActivationFunction(Derivable):
    """
    Base class of an activation function.
    """


class Sigmoid(ActivationFunction):
    """
    Sigmoid function.

    x => 1 / (1 + e^-x)
    """
    def __call__(self, x):
        return (np.ones(x.shape) + np.exp(-x))**-1

    def gradient(self, x):
        return self(x)*(1-self(x))


class Identity(ActivationFunction):
    """
    Identity function.

    x => x
    """
    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class ReLU(ActivationFunction):
    """
    ReLU

    x => x       if x > 0
         0       if x < 0
    """
    def __call__(self, x):
        return np.where(x > 0, x, 0)

    def gradient(self, x):
        return np.where(x > 0, 1, 0)


class LeakyReLU(ActivationFunction):
    """
    Leaky ReLU

    x => x       if x > 0
         x*gamma if x < 0
    """
    def __init__(self, gamma=.01):
        self.gamma = gamma

    def __call__(self, x):
        return np.where(x > 0, x, x*self.gamma)

    def gradient(self, x):
        return np.where(x > 0, 1, self.gamma)


class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent

    x => tanh(x)
    """
    def __call__(self, x):
        return np.tanh(x)

    def gradient(self, x):
        return 1 - np.tanh(x)**2


class HardTanh(ActivationFunction):
    """
    Hard Hyperbolic Tangent

    x => tanh(x)
    """
    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, x):
        x = np.where(x < self.min, self.min, x)
        return np.where(x > self.max, self.max, x)

    def gradient(self, x):
        x = np.where(x < self.min, 1, x)
        return np.where(x > self.max, 1, x)


class BipolarSigmoid(ActivationFunction):
    """
    Generic Sigmoid Function

    x => (a - b)/(1 + e^d(c/2-x)) + b

    (From here:https://stackoverflow.com/questions/43213069/fit-bipolar-sigmoid-python)
    """
    def __init__(self, a, b, c, d):
        """
        :param a: max_height
        :param b: min_height
        :param c: end_slope
        :param d: slope
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x):
        return ((self.a-self.b) / (1 + np.exp(self.c/2-x)**self.d)) + self.b

    def gradient(self, x):
        return (-(self.a-self.b) * (1 + np.exp(self.c/2-x)**self.d)**-2)*np.exp(self.c/2-x)**self.d * self.d


class SoftPlus(ActivationFunction):
    """
    SoftPlus

    x => ln(1+e^-x)
    """

    def __call__(self, x):
        return np.log(np.ones(x.shape) + np.exp(-x))

    def gradient(self, x):
        return (np.ones(x.shape) + np.exp(-x))**-1


class ExpLinear(ActivationFunction):
    """
    Exponential to Linear

    x => x              if x > 0
         gamma*(e^-x-1) if x > 0

    """
    def __init__(self, gamma):
        """
        :param gamma: slope
        """
        self.gamma = gamma

    def __call__(self, x):
        return np.where(x < 0, self.gamma*(np.exp(x) - 1), x)

    def gradient(self, x):
        return np.where(x < 0, self.gamma*np.exp(x), 1)
