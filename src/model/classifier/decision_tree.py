from src.model.classifier.logistic import Logistic
from src.model.model import Model
import numpy as np


class DecisionTree(Model):
    def __init__(self, n_features):
        self.logistic_models = [Logistic(1) for i in range(n_features)]
        self.n_features = n_features

    def __call__(self, x):
        indiv_pred = [model(x[:, i]) for i, model in zip(range(self.n_features), self.logistic_models)]

    def get_params(self):
        return [model.get_params() for model in self.logistic_models]
