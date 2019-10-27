from src.optimization.linear import *
from src.model.classifier.logistic import Logistic
from src.functions.distance import L2
from src.utils.data_manipulation import *
from src.optimization.logistic import LogisticSGD
from src.functions.loss import MAE
from src.preconditioning.feature_filling import MeanFilling
from src.preconditioning.normalization import MinMaxNormalizer
import os


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    idx = ~np.logical_or(np.any(data == -999, axis=1), np.any(data == 0, axis=1))
    data[:, 24] /= 3

    new_data = MeanFilling(data[:, 2:])(data[:, 2:])
    new_data = MinMaxNormalizer()(np.hstack([np.reshape(data[:, 1], (-1, 1)),
                                  new_data[:, [4, 5, 12, 27, 28, 0]],
                                  # new_data[:, [4, 5, 12, 27, 28, 0]] ** 2,
                                  np.ones((new_data.shape[0], 1))]))

    # train_1, test = split(data[idx], test_relative_size=.5)
    train, test = split(new_data[~idx], test_relative_size=.25)
    model = Logistic(new_data.shape[1]-1)
    optimizer = LogisticSGD()
    lr = 10**-1
    step = .9
    for i in range(30):
        model.set_param(np.random.normal(0, 1, (new_data.shape[1]-1,1)))
        prediction = np.where(model(test[:, 1:]) > 0.5, 1, 0)
        expected = np.reshape(test[:, 0], (-1, 1))
        error = np.abs(prediction - expected)
        error = np.sum(error) / error.size
        print(error)

        optimizer(model=model, tx=train[:, 1:], y=np.reshape(train[:, 1], (-1, 1)),
                  lr=lr, batch_size=10, batch_number=1000, epochs=200, regularization=2,
                  epoch_step=(200, .9))
        optimizer(model=model, tx=train[:, 1:], y=np.reshape(train[:, 1], (-1, 1)),
                  lr=lr/2, batch_size=10, batch_number=1000, epochs=200, regularization=2,
                  epoch_step=(200, .9))
        optimizer(model=model, tx=train[:, 1:], y=np.reshape(train[:, 1], (-1, 1)),
                  lr=lr/4, batch_size=10, batch_number=1000, epochs=200, regularization=2,
                  epoch_step=(200, .9))
        optimizer(model=model, tx=train[:, 1:], y=np.reshape(train[:, 1], (-1, 1)),
                  lr=lr/8, batch_size=10, batch_number=1000, epochs=200, regularization=2,
                  epoch_step=(200, .9))
        optimizer(model=model, tx=train[:, 1:], y=np.reshape(train[:, 1], (-1, 1)),
                  lr=lr/16, batch_size=10, batch_number=1000, epochs=200, regularization=2,
                  epoch_step=(200, .9))


