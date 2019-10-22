from src.model.nn.nn_layer import NNLayer
from src.optimization.optimizer import Optimizer
from src.functions.loss import MSE
import numpy as np
from src.utils.data_manipulation import batch_iter


class NNLayerSGD(Optimizer):
    def __call__(self, model: NNLayer, tx, y, **kwargs):
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

        activation = model.get_activation_function()

        for epoch in range(epochs):
            running_loss = 0
            for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches):
                txw = np.dot(batch_tx, model.get_params())
                model.set_param(model.get_params() - lr * np.dot(np.transpose(batch_tx, (1, 0)),
                                                                 loss.gradient(activation(batch_tx), batch_y) *
                                                                 activation.gradient(batch_tx)))
                print(loss(activation(txw), batch_y))
            print(running_loss)


class NNLayerGD(Optimizer):
    def __call__(self,  model: NNLayer, tx, y, **kwargs):
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

        activation = model.get_activation_function()

        for i in range(epochs):
            txw = np.dot(tx, model.get_params())
            loss_rad = loss.gradient(activation(txw), y)
            act_grad = activation.gradient(txw)
            grad = lr * np.dot(np.transpose(tx, (1, 0)),
                               loss.gradient(activation(txw), y) *
                               activation.gradient(txw))
            new_w = model.get_params() - grad
            model.set_param(model.get_params() - lr * np.dot(np.transpose(tx, (1, 0)),
                                                             loss.gradient(activation(txw), y) *
                                                             activation.gradient(txw)))
            print(loss(activation(txw), y))
