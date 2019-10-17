from model.nn.nn_layer import NNLayer
from src.functions.activation_functions import Sigmoid
import numpy as np


class Logistic(NNLayer):
    def __init__(self, input_size, activation: Sigmoid()):
        super().__init__((input_size, 1), activation_function=activation)

    def __call__(self, tx):
        return super().__call__(tx)

