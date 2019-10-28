from src.optimization.optimizer import Optimizer
from src.utils.data_manipulation import *
from src.functions.loss import LogCosh
from src.model.regression.linear_model import LinearModel
import sys

class LinearOptimizer(Optimizer):
    """
    Optimizer of linear models.
    """


class LinearSGD(LinearOptimizer):
    def __str__(self):
        return "LinearSGD"

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
        num_batches = min(kwargs['num_batches'], tx.shape[0]) if 'num_batches' in kwargs else 1000
        loss = kwargs['loss'] if 'loss' in kwargs else LogCosh()
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        epoch_step = kwargs['epoch_step'] if 'epoch_step' in kwargs else (50, 0.75)

        i = 0
        running_loss = 0
        for step in range(int(epochs/epoch_step[0])):
            for epoch_iter in range(epoch_step[0]):
                running_loss = 0
                acc_grad = 0

                for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches):
                    out = model(batch_tx)
                    running_loss += loss(out, batch_y)
                    grad = np.dot(np.transpose(batch_tx, (1, 0)), loss.gradient(model(batch_tx), batch_y))
                    model.set_param(model.get_params() - lr * grad)
                    acc_grad += np.sum(np.abs(grad))

                if acc_grad < lr * 10 ** -2 / model.get_params().size:
                    return
            i += 1
            lr *= epoch_step[1]


class LinearGD(LinearOptimizer):
    def __str__(self):
        return "LinearGD"

    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Stochastic Gradient Descent.
        :param tx: sample
        :param y: labels
        :param max_iter: number of batches to learn
        :param loss: loss function
        :param lr: learning rate
        """
        loss = kwargs['loss'] if 'loss' in kwargs else LogCosh()
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100

        for epoch in range(epochs):
            out = model(tx)
            # print(loss(out, y))
            gradient = np.dot(np.transpose(tx, (1, 0)), loss.gradient(model(tx), y))
            model.set_param(model.get_params() - lr * gradient)

            if np.sum(np.abs(gradient)) < lr*10**-2/model.get_params().size:
                break


class Ridge(LinearOptimizer):
    def __str__(self):
        return "Ridge"

    def __call__(self, model: LinearModel, tx,  y, **kwargs):
        """
        Performs Ridge regression.
        :param tx: sample
        :param y: labels
        :param lambda_: ridge hyper-parameter
        """
        lambda_ = kwargs['lambda_'] if 'lambda_' in kwargs else 10**-5

        #w=(XT*X+lambda*I)^-1*XT*y
        w = np.linalg.inv(np.transpose(tx)@tx + lambda_/(2*len(y))*np.eye(tx.shape[1], tx.shape[1])) @np.transpose(tx) @y
        model.set_param(w)


class Lasso(LinearOptimizer):
    def __str__(self):
        return "Lasso"


class LS(LinearOptimizer):
    def __str__(self):
        return "LS"

    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Least Squares
        :param tx: sample
        :param y: labels
        """
        #w=(XT*X)^-1*ATy
        w=np.linalg.inv(tx.transpose()@tx)@tx.transpose()@y
        model.set_param(w)


class OLS(LinearOptimizer):
    def __call__(self, model: LinearModel, tx, y, **kwargs):
        """
        Performs Ordinary Least Squares
        :param tx: sample
        :param y: labels
        """
        pass
