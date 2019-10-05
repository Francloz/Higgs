import numpy as np
from src.utils.scheduler import *

class NeuronLayer:
    def __init__(self, input_size, output_size, activation_function, initial_weights=None, hidden=True):
        """
        Neural network layer constructor.
        :param input_size: input size
        :param output_size: output size
        :param activation_function: activation function
        :param initial_weights: initial weights (output_size, input_size+1)
        """
        self.weights = np.random.normal(0, 1, (output_size, input_size+1)) if not initial_weights else initial_weights
        self.act_fun = activation_function
        self.last_input = np.zeros(input_size)
        self.last_output = np.zeros(output_size)
        self.hidden = hidden

    def __call__(self, inputs):
        """
        Processes the input and returns the output.

        w0,0  ... w0,n w0,n+1     x1       y1
        .          .      .        .        .
        .          .      .    x   .   =    .
        .          .      .        .        .
        wk,0  ... wkn  wk,n+1     xn       yk
                                   1

        :param inputs: input of the layer
        :return: output of the layer
        """
        self.last_input = np.append(inputs, np.ones((1, 1)), axis=0)

        self.last_output = np.dot(self.weights, self.last_input)
        output = self.act_fun(self.last_output)
        return output

    def back_prop(self, param, lr=10**-3):
        """
        Optimizes the NN with the optimizer.
        :param param: parameter of the back-propagation that is either the returned back_prop of the next layer or the
        target output
        :param lr: learning rate
        :return: returns the parameter that must be passed to the previous layer
        """
        if not self.hidden:

            delta = np.dot(self.act_fun(self.last_output) - param, self.act_fun.derivative(self.last_output))
        else:
            weight = param[:-1]
            derivative = self.act_fun.derivative(self.last_output)
            # delta = np.dot(param[:-1], self.act_fun.derivative(self.last_output))
            delta = param[:-1] * self.act_fun.derivative(self.last_output)
        err = np.dot(delta, np.transpose(self.last_input, (1, 0)))
        self.weights -= (lr() if isinstance(lr, Scheduler) else lr)*err
        return np.dot(np.transpose(self.weights, (1, 0)), delta)


class NeuralNetwork:
    def __init__(self, layers):
        """
        Neural Network constructor.
        :param layers:
        """
        if sum(not isinstance(l, NeuronLayer) for l in layers) > 0:
            raise ValueError("The layers are not neural network layers.")
        self.layers = layers

    def __call__(self, inputs):
        """
        Processes the input and returns the output.
        :param inputs: input of the nn
        :return: output of the nn
        """
        inputs = np.expand_dims(inputs, axis=1)
        for i in range(len(self.layers)):
            inputs = self.layers[i](inputs)
        return inputs

    def back_prop(self, target, lr=10**-1):
        """
        Optimizes the NN with the optimizer.
        :param optimizer:
        :return:
        """
        for i in range(len(self.layers)-1, -1, -1):
            target = self.layers[i].back_prop(target, lr)






