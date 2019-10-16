from model.nn.nn_layer import NNLayer
from src.optimization.optimizer import Optimizer
from src.functions.loss import MSE
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


class NNLayerGD(Optimizer):
    def __init__(self, model: NNLayer):
        super().__init__(model)

    def __call__(self, tx, y, **kwargs):
        """
        Performs Gradient Descent.
        :param tx: sample
        :param y: labels
        :param epochs: number of timesto go through the dataset
        :param loss: loss function
        :param lr: learning rate
        """
        loss = kwargs['loss'] if 'loss' in kwargs else MSE()
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1000

        activation = self.model.get_activation_function()

        for i in range(epochs):
            txw = np.dot(tx, self.model.get_w())
            loss_grad = loss.gradient(activation(txw), y)
            act_grad = activation.gradient(txw)
            # (2, 4) x (4 x 2) x (2 x 4)
            self.model.set_param(self.model.get_w() - lr * np.dot(np.transpose(tx, (1, 0)),
                                                                  loss.gradient(activation(txw), y) *
                                                                  activation.gradient(txw)))
            print(loss(activation(txw), y))
