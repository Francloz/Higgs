from src.model.regression.linear_model import LinearModel
from src.functions.activation_functions import Sigmoid
import numpy as np


class Logistic(LinearModel):
    def __init__(self, input_size):
        super().__init__((input_size, 1))

    def __call__(self, tx):
        return Sigmoid()(tx.dot(self.w))

