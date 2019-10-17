from src.model.regression.linear_model import *
from src.optimization.linear import *
from src.optimization.logistic import *
from src.model.nn.nn_layer import *
from src.functions.activation_functions import *
from src.model.classifier.logistic import Logistic

if __name__ == "__main__":
    model = LinearModel((2, 2))
    x = np.array([[1, 2],
                  [2, 3],
                  [0, 0],
                  [-1, -1]], dtype=np.double)
    y = np.array([[1, 2],
                  [2, 3],
                  [0, 0],
                  [-1, -1]], dtype=np.double)
    loss = LogCosh()
    optimizer = LinearGD()
    optimizer(model, x, y, lr=0.01, num_batches=100, loss=loss, batch_size=1, epochs=0)
    print(model.get_w())

    model = NNLayer((2, 2), Identity())
    model.set_param(np.ones((2, 2)))
    optimizer = NNLayerGD()
    optimizer(model, x, y, lr=0.01, num_batches=100, loss=loss, batch_size=1, epochs=0)
    print(model.get_w())

    optimizer = LogisticGD()
    """
    :param a: max_height
    :param b: min_height
    :param c: end_slope
    :param d: slope
    """
    model = Logistic(3, Sigmoid())
    model.set_param(np.ones((3, 1)))
    x = np.array([[1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 0]], dtype=np.double)
    y = np.array([[.1],
                  [.9],
                  [.9],
                  [.1]], dtype=np.double)
    new_tx = np.ones((x.shape[0], x.shape[1] + 1))
    new_tx[:, :-1] = x
    optimizer(model, new_tx, y, lr=0.01, num_batches=100, loss=loss, batch_size=1, epochs=10000)
    print(model(new_tx))
