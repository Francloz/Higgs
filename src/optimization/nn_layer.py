from src.model.nn_layer import NNLayer
from src.optimization.optimizer import Optimizer
from src.functions.loss import MSE
from src.functions.activation_functions import ActivationFunction
import numpy as np

class NNLayerSGD(Optimizer):
    def __init__(self, model: NNLayer):
        super().__init__(model)

    def __call__(self, tx, y, **kwargs):
        """
        Optimizes the model.
        :param tx: sample
        :param y: labels
        """
        w = self.model.get_param()


class NNLayerGD(Optimizer):
    def __init__(self, model: NNLayer):
        super().__init__(model)

    def __call__(self, tx, y, **kwargs):
        """
        Performs Gradient Descent.
        :param tx: sample
        :param y: labels
        :param max_iter: number of batches to learn
        :param loss: loss function
        :param lr: learning rate
        """
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1
        loss = kwargs['loss'] if 'loss' in kwargs else MSE()
        lr = kwargs['lr'] if 'lr' in kwargs else .01

        w = self.model.get_param()
        activation = self.model.get_activation_function()

        for i in range(max_iter):
            txw = np.dot(tx, w)
            w -= lr * loss.gradient(activation(txw, np.ones()), y, np.ones(w.shape)) * activation.derivative(txw)
