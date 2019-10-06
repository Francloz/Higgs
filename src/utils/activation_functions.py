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

    x => 1 / (1 + e^-x)
    """
    def __call__(self, x):
        return (np.ones(x.shape) + np.exp(-x))**-1

    def derivative(self, x):
        return self(x)*(1-self(x))


class Identity(ActivationFunction):
    """
    Identity function.

    x => x
    """
    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class ReLU(ActivationFunction):
    """
    ReLU

    x => x       if x > 0
         0       if x < 0
    """
    def __call__(self, x):
        return np.where(x > 0, x, 0)

    def derivative(self, x):
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

    def derivative(self, x):
        return np.where(x > 0, 1, self.gamma)


class Tanh(ActivationFunction):
    """
    Hyperbolic Tangent

    x => tanh(x)
    """
    def __call__(self, x):
        return np.tanh(x)

    def derivative(self, x):
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

    def derivative(self, x):
        x = np.where(x < self.min, 0, x)
        return np.where(x > self.max, 0, x)


class BipolarSigmoid(ActivationFunction):
    """
    Generic Sigmoid Function

    x => (a - b)/(1 + e^d(x - c/2)) + b
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
        return ((self.a-self.b) / (1 + np.exp(x-(self.c/2))**self.d)) + self.b

    def derivative(self, x):
        return (-(self.a-self.b) * (1 + np.exp(x-(self.c/2))**self.d)**-2)*np.exp(x-(self.c/2))**self.d * self.d
