from model.model import Model
import numpy as np
from utils.data_manipulation import *
import time

"""
This file will be deprecated soon.
"""


class ModelManager:
    def __init__(self, model: Model, directory: str):
        self.model = model
        self.dir = directory

    def _train_and_test(self, tx_tr, y_tr, tx_te, y_te, loss, lr, max_iter, batch_size):
        self.model.learn(tx_tr, y_tr, loss, lr, max_iter, batch_size)
        return self.model.evaluate(tx_te, y_te, loss)

    def learn(self, tx, y, loss, test=.3, epochs=100, max_iter=100, lr=.01, batch_size=1):
        tx_tr, y_tr, tx_te, y_te = separate(tx, y, test)
        running_loss = np.zeros(epochs)
        for epoch in range(epochs):
            running_loss[epoch] = self.train_and_test(tx_tr, y_tr, tx_te, y_te, loss, max_iter, lr, batch_size)
            self.model.save_model(self.dir + time.strftime("%Y.%m.%d_%H.%M.%S"))

