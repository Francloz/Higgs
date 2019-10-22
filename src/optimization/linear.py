from src.optimization.optimizer import Optimizer
from src.utils.data_manipulation import *
from src.functions.loss import MSE
from src.model.regression.linear_model import LinearModel


class LinearOptimizer(Optimizer):
    """
    Optimizer of linear models.
    """


class LinearSGD(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Stochastic Gradient Descent.
        :param tx: sample
        :param y: labels
        :param batch_size: size of the batches
        :param num_batches: number of batches to learn
        :param loss: loss function
        :param lr: learning rate
        :param epoch: number of times to go over the dataset
        """
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 1
        num_batches = min(kwargs['num_batches'], tx.shape[0]) if 'num_batches' in kwargs else 1
        loss = kwargs['loss'] if 'loss' in kwargs else MSE()
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100

        for epoch in range(epochs):
            running_loss = 0
            for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches):
                loss_grad = loss.gradient(model(batch_tx), batch_y)
                x = np.transpose(batch_tx, (1, 0))
                out = model(batch_tx)
                running_loss += loss(out, y)
                model.set_param(model.get_params() - lr * np.dot(np.transpose(batch_tx, (1, 0)),
                                                                 loss.gradient(model(batch_tx), batch_y)))
            print(running_loss)


class LinearGD(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Stochastic Gradient Descent.
        :param tx: sample
        :param y: labels
        :param max_iter: number of batches to learn
        :param loss: loss function
        :param lr: learning rate
        """
        loss = kwargs['loss'] if 'loss' in kwargs else MSE()
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100

        for epoch in range(epochs):
            running_loss = 0
            loss_grad = loss.gradient(model(tx), y)
            x = np.transpose(tx, (1, 0))
            out = model(tx)
            print(loss(out, y))
            model.set_param(model.get_params() - lr * np.dot(np.transpose(tx, (1, 0)),
                                                             loss.gradient(model(tx), y)))


class Ridge(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Ridge regression.
        :param tx: sample
        :param y: labels
        :param lambda_: ridge hyper-parameter
        """
        pass


class Lasso(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Lasso regression.
        :param tx: sample
        :param y: labels
        :param lambda_: lasso hyper-parameter
        """
        pass


class LS(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Least Squares
        :param tx: sample
        :param y: labels
        """
        pass


class OLS(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Ordinary Least Squares
        :param tx: sample
        :param y: labels
        """
        pass
