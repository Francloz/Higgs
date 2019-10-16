from functions.loss import *
from src.model.regression.linear_model import *
from src.optimization.linear import *
from src.optimization.nn_layer import *
from src.model.nn.nn_layer import *

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
    optimizer = GD()
    optimizer(model, x, y, lr=0.01, num_batches=100, loss=loss, batch_size=1, epochs=10000)
    print(model.get_w())

    model = NNLayer((2, 2), Identity())
    optimizer = NNLayerGD()
    optimizer(model, x, y, lr=0.01, num_batches=100, loss=loss, batch_size=1, epochs=10000)
    print(model.get_w())
