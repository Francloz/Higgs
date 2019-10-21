from src.model.model import Model
from src.utils.data_manipulation import *
from src.optimization.optimizer import Optimizer
from src.functions.loss import Loss
import time

"""
This file will be deprecated soon.
"""


class ModelManager:
    def __init__(self, model: Model, optimizer: Optimizer, loss : Loss,  directory: str):
        self.model = model
        self.dir = directory
        self.optimizer = optimizer
        self.loss = loss

    def _train_and_test(self, tx_tr, y_tr, tx_te, y_te, **kwargs):
        self.optimizer(self.model, tx_tr, y_tr, kwargs)
        return self.loss(self.model(tx_te), y_te)

    def learn(self, tx, y, loss, test=.3, **kwargs):
        tx_tr, y_tr, tx_te, y_te = separate(tx, y, test)
        self._train_and_test(tx_tr, y_tr, tx_te, y_te, kwargs)
        self.model.save_model(self.dir + time.strftime("%Y.%m.%d_%H.%M.%S"))

