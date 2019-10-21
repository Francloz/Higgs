from src.optimization.optimizer import Optimizer
from src.model.classifier.logistic import Logistic
from src.utils.data_manipulation import batch_iter
from src.functions.activation_functions import Sigmoid

# from src.functions.loss import MSE


class LogisticGD(Optimizer):
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
            model.set_param(w - lr*(tx.T.dot(sigma(tx.dot(w))-y) - regularize*w))
            # print(MSE()(sigma(tx.dot(w)), y))


class LogisticSGD(Optimizer):
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
        num_batches = min(kwargs['num_batches'], tx.shape[0]) if 'num_batches' in kwargs else 1
        lr = kwargs['lr'] if 'lr' in kwargs else .01
        epochs = kwargs['epochs'] if 'epochs' in kwargs else 100
        regularize = kwargs['regularize'] if 'regularize' in kwargs else 0
        sigma = Sigmoid()
        for epoch in range(epochs):
            for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches):
                w = model.get_params()
                model.set_param(w - lr*(batch_tx.T.dot(sigma(batch_tx.dot(w))-batch_y) + regularize*w))
            # print(MSE()(sigma(tx.dot(model.get_w())), y))


