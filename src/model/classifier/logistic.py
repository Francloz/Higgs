from model.nn.nn_layer import NNLayer
from src.functions.activation_functions import Sigmoid


class Logistic(NNLayer):
    def __init__(self, input_size):
        super().__init__((input_size+1, 1), Sigmoid())


