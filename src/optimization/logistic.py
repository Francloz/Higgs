from src.optimization.optimizer import Optimizer
from src.model.classifier.logistic import Logistic
from src.utils.data_manipulation import batch_iter
from src.functions.activation_functions import Sigmoid
from src.functions.loss import MAE
# from src.functions.loss import MSE
import numpy as np

class LogisticGD(Optimizer):
    def __str__(self):
        return "LogisticGD"

    def __call__(self, model: Logistic, tx, y, **kwargs):
        """
        Performs Gradient Descent.
        :param tx: sample
        :param y: labels
        :param epochs: number of timesto go through the dataset
        :param loss: loss function
        :param lr: learning rate
        """
        lr = kwargs['lr'] if 'lr' in kwargs else .1
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 1000
        regularize = kwargs['regularize'] if 'regularize' in kwargs else 0.1
        sigma = Sigmoid()
        for epoch in range(epochs):
            w = model.get_params()
            grad = tx.T.dot(sigma(tx.dot(w))-y) - regularize*w
            model.set_param(w - lr*grad)
            # print(MSE()(sigma(tx.dot(w)), y))
            if np.sum(np.abs(grad)) < lr * 10 ** -2 / model.get_params().size:
                return


class LogisticSGD(Optimizer):
    def __str__(self):
        return "LogisticSGD"

    def __call__(self, model: Logistic, tx, y, **kwargs):
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
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        epoch_step = kwargs['epoch_step'] if 'epoch_step' in kwargs else (50, 0.75)
        regularize = kwargs['regularize'] if 'regularize' in kwargs else 0
        sigma = Sigmoid()

        for step in range(int(epochs/epoch_step[0])):
            for epochs in range(epoch_step[0]):
                running_loss = 0
                acc_grad = 0

                for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches):
                    w = model.get_params()
                    grad = (batch_tx.T.dot(sigma(batch_tx.dot(w))-batch_y) + regularize*w)
                    acc_grad += np.sum(np.abs(grad))
                    model.set_param(w - lr*grad)
                    running_loss += MAE()(model(batch_tx), batch_y)
                # print(running_loss/batch_size/num_batches)

                if acc_grad < lr * 10 ** -2 / model.get_params().size:
                    return

            lr *= epoch_step[1]


