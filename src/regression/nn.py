import numpy as np


class NeuronLayer:
    def __init__(self, input_size, output_size, activation_function, initial_weights=None):
        self.weights = np.random.normal(0, 1, (input_size, output_size)) if not initial_weights else initial_weights
        self.act_fun = activation_function
        self.last_input = np.zeros(input_size)
        self.last_output = np.zeros(output_size)

    def __call__(self, inputs):
        self.last_input = inputs
        self.last_output = np.dot(self.weights, inputs)
        return self.act_fun(self.last_output)

    def optimize(self, optimizer):
        self.weights = optimizer(self.last_input, self.last_output, self.act_fun, self.weights)


class NeuralNetwork:
    def __init__(self, layers):
        if sum(not isinstance(l,NeuronLayer) for l in layers) > 0:
            raise ValueError("The layers are not neural layers.")
        self.layers = layers

    def __call__(self, inputs):
        i

