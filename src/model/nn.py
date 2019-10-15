import numpy as np
import os
from pathlib import Path
from src.utils.scheduler import *

"""
This file will be deprecated in future commits.
"""


class NeuronLayer:
    def __init__(self, input_size, output_size, activation_function, initial_weights=None, hidden=True):
        """
        Neural network layer constructor.
        :param input_size: input size
        :param output_size: output size
        :param activation_function: activation function
        :param initial_weights: initial weights (output_size, input_size+1)
        """
        self.weights = np.random.normal(0, .1, (output_size, input_size+1)) if not initial_weights else initial_weights
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
        self.last_input = np.append(inputs, np.ones((1, inputs.shape[1])), axis=0)
        self.last_output = np.dot(self.weights, self.last_input)
        output = self.act_fun(self.last_output)
        return output

    def back_prop(self, param, lr):
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
            delta = param[:-1] * self.act_fun.derivative(self.last_output)
        err = np.dot(delta, np.transpose(self.last_input, (1, 0)))
        self.weights -= (lr() if isinstance(lr, Scheduler) else lr)*err
        return np.dot(np.transpose(self.weights, (1, 0)), delta)

    def save(self, path):
        f = open(path, "a")
        np.savetxt(f, self.weights)
        f.close()

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights


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

    def _back_prop(self, target, lr):
        """
        Optimizes the NN with the optimizer.
        :param optimizer:
        :return:
        """
        for i in range(len(self.layers)-1, -1, -1):
            target = self.layers[i].back_prop(target, lr)

    def learn(self, inputs, labels, lr=10**-3):
        for col in range(inputs.shape[0]):
            output = self(inputs[col, :])
            self._back_prop(labels[col, :], lr)

    def save(self, path):
        f = open(path, "w+")
        f.close()

        parameter_file = Path(path)
        if parameter_file.is_file():
            os.remove(path)
        for layer in self.layers:
            layer.save(path)

    def load(self, path):
        file_array = np.fromfile(path, sep=" ")
        for layer in self.layers:
            layer.set_weights(np.reshape(file_array[:layer.get_weights().size], layer.get_weights().shape))


"""
import numpy as np
from src.utils.scheduler import *
from src.utils.error import *


class NeuronLayer:
    def __init__(self, input_size, output_size, activation_function, initial_weights=None, hidden=True):
        ""
        Neural network layer constructor.
        :param input_size: input size
        :param output_size: output size
        :param activation_function: activation function
        :param initial_weights: initial weights (output_size, input_size+1)
        ""
        self.weights = np.random.normal(0, 1, (output_size, input_size+1)) if not initial_weights else initial_weights
        self.act_fun = activation_function
        self.last_input = []
        self.last_output = []
        self.hidden = hidden
        self.shape = (input_size, output_size)

    def __call__(self, inputs):
        ""
        Processes the input and returns the output.

        w0,0  ... w0,n w0,n+1     x1       y1
        .          .      .        .        .
        .          .      .    x   .   =    .
        .          .      .        .        .
        wk,0  ... wkn  wk,n+1     xn       yk
                                   1

        :param inputs: input of the layer
        :return: output of the layer
        ""
        #print("(%d,%d) (%d,%d)" % (inputs.shape[0], inputs.shape[1], 1, inputs.shape[1]))
        self.last_input = np.append(inputs, np.ones((1, inputs.shape[1])), axis=0)
        self.last_output = np.dot(self.weights, self.last_input)
        output = self.act_fun(self.last_output)
        return output

    def back_prop(self, param, lr=10**-3):
        ""
        Optimizes the layer with the back propagation. Returns the parameter to be passed to the previous layer.
        :param param: target value to the last layer, the return of this function to the previous layer
        :param lr: learning rate, can be a Scheduler
        ""
        ret = np.zeros((self.shape[0], param.shape[1]))

        for col in range(param.shape[1]):
            if not self.hidden:
                param[:, col] = self.act_fun(self.last_output[:, col]) - param[:, col]
                delta = np.dot(param[:, col], self.act_fun.derivative(self.last_output))
            else:
                delta = param[:-1, col] * self.act_fun.derivative(self.last_output[:, col])

            delta = np.expand_dims(delta, axis=0)
            err_der = np.dot(delta, np.expand_dims(self.last_input[:, col], axis=0))
            self.weights -= (lr() if isinstance(lr, Scheduler) else lr)*err_der
            ret[:, col] = np.dot(np.transpose(self.weights, (1, 0)), delta)[:-1]
        return ret


class NeuralNetwork:
    def __init__(self, layers):
        ""
        Neural Network constructor.
        :param layers:
        ""
        if sum(not isinstance(l, NeuronLayer) for l in layers) > 0:
            raise ValueError("The layers are not neural network layers.")
        self.layers = layers

    def __call__(self, inputs):
        ""
        Processes the input and returns the output.
        :param inputs: input of the nn
        :return: output of the nn
        ""
        for i in range(len(self.layers)):
            inputs = self.layers[i](inputs)
        return inputs

    def learn(self, inputs, labels, loss=mse, lr=10**-3):
        ""
        Optimizes the NN with the optimizer.
        :param inputs: inputs
        :param labels: target labels
        :param loss: loss function
        :param lr: learning rate, can be an scheduler
        ""
        output = self(inputs)
        for i in range(len(self.layers)-1, -1, -1):
            labels = self.layers[i].back_prop(labels, lr)
        return loss(output, labels)
"""



