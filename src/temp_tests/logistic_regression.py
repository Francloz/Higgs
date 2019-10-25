from src.optimization.linear import *
from src.model.classifier.logistic import Logistic
from src.functions.distance import L2
from src.utils.data_manipulation import *
from src.optimization.logistic import LogisticSGD
from src.functions.loss import MAE
import os


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    idx = ~np.logical_or(np.any(data == -999, axis=1), np.any(data == 0, axis=1))
    data[:, 2:] = np.load(file=path + 'filled_dataset.npy')
    data[:, 24] /= 3

    # train_1, test = split(data[idx], test_relative_size=.5)
    train, test = split(np.vstack([data[~idx]]), test_relative_size=.25)

    model = Logistic(31)
    optimizer = LogisticSGD()
    lr = 10**-1
    step = .9
    for i in range(30):
        optimizer(model=model, tx=np.hstack([train[:, 2:], np.ones((train.shape[0], 1))]), y=np.reshape(train[:, 1], (-1,1)),
                  lr=lr, batch_size=50, batch_number=10000, epochs=25, regularization=2)
        lr *= step

        prediction = np.where(model(np.hstack([test[:, 2:], np.ones((test.shape[0], 1))])) > 0.5, 1, 0)
        expected = np.reshape(test[:, 1], (-1, 1))
        error = np.abs(prediction - expected)
        error = np.sum(error) / error.size
        print(error)
