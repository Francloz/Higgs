from optimization.optimizer import Optimizer
from utils.data_manipulation import *
from functions.loss import *
from model.regression.linear_model import LinearModel


class LinearOptimizer(Optimizer):
    def __init__(self, model: LinearModel):
        super().__init__(model)


class SGD(LinearOptimizer):
    def __call__(self, tx, y, **kwargs):
        """
        Performs Stochastic Gradient Descent.
        :param tx: sample
        :param y: labels
        :param batch_size: size of the batches
        :param num_batches: number of batches to learn
        :param loss: loss function
        :param lr: learning rate
        """
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
        num_batches = kwargs['num_batches'] if 'num_batches' in kwargs else 1
        loss = kwargs['loss'] if 'loss' in kwargs else MSE()
        lr = kwargs['lr'] if 'lr' in kwargs else .01

        for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches):
            self.w -= lr * loss.gradient(batch_tx, batch_y, self.w)


class GD(LinearOptimizer):
    def __call__(self, tx, y, **kwargs):
        """
        Performs Stochastic Gradient Descent.
        :param tx: sample
        :param y: labels
        :param max_iter: number of batches to learn
        :param loss: loss function
        :param lr: learning rate
        """
        max_iter = kwargs['max_iter'] if 'max_iter' in kwargs else 1
        loss = kwargs['loss'] if 'loss' in kwargs else MSE()
        lr = kwargs['lr'] if 'lr' in kwargs else .01

        for i in range(max_iter):
            self.w -= lr * loss.gradient(tx, y, self.model.get_w())


class Ridge(LinearOptimizer):
    def __call__(self, tx, y, **kwargs):
        """
        Performs Ridge regression.
        :param tx: sample
        :param y: labels
        :param lambda_: ridge hyper-parameter
        """
        pass


class Lasso(LinearOptimizer):
    def __call__(self, tx, y, **kwargs):
        """
        Performs Lasso regression.
        :param tx: sample
        :param y: labels
        :param lambda_: lasso hyper-parameter
        """
        pass


class LS(LinearOptimizer):
    def __call__(self, tx, y, **kwargs):
        """
        Performs Least Squares
        :param tx: sample
        :param y: labels
        """
        pass


class OLS(LinearOptimizer):
    def __call__(self, tx, y, **kwargs):
        """
        Performs Ordinary Least Squares
        :param tx: sample
        :param y: labels
        """
        pass
