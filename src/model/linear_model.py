from utils.data_manipulation import *
from functions.loss import MSE


class LinearModel:
    def __init__(self, shape):
        self.w = np.zeros(shape)

    def __call__(self, tx):
        return np.dot(tx, self.w)

    def set_param(self, w):
        self.w = w

    def load_model(self, path):
        self.w = np.reshape(np.fromfile(path, sep=" ")[:self.w.size], self.w.shape)

    def save_model(self, path):
        f = open(path, "w+")
        np.savetxt(f, self.w)
        f.close()

    def learn(self, tx, y, max_iter=1000, loss=MSE, batch_size=1, lr=0.01):
        for i in range(max_iter):
            for batch_y, batch_tx in batch_iter(y, tx, batch_size):
                self.w -= lr*loss.gradient(batch_tx, batch_y, self.w)

    def evaluate(self, tx, y, loss):
        return loss(tx, y, self.w)
