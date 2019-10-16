from model.classifier.classifier import Classifier
from model.nn_layer import NNLayer
from src.functions.activation_functions import Sigmoid
import numpy as np


class Logistic(NNLayer):
    def __init__(self, input_size):
        super().__init__((input_size+1, 1), Sigmoid())


