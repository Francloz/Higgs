from src.optimization.nn_layer import *


class LogisticGD(NNLayerGD):
    def __call__(self, model: NNLayer, tx, y, **kwargs):
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
        new_tx = np.ones((tx.shape[0], tx.shape[1] + 1))
        new_tx[:, :-1] = tx
        super().__call__(model, new_tx, y, loss=loss, lr=lr, epochs=epochs)


class LogisticSGD(NNLayerSGD):
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
        new_tx = np.ones((tx.shape[0], tx.shape[1] + 1))
        new_tx[:, :-1] = tx
        super().__call__(model, new_tx, y, batch_size=batch_size, num_batches=num_batches,
                         loss=loss, epochs=epochs, lr=lr)
