from src.regression.nn import *
from src.utils.activation_functions import *
import sys

if __name__ == "__main__":

    nn = NeuralNetwork([NeuronLayer(1, 1, Identity(), hidden=False)])
    max_iter = 1000
    for i in range(max_iter):
        sys.stdout.write("\rProgress: %d%%" % int(i/max_iter*100))
        nn.learn(np.array([[0], [1], [2], [3]]), np.array([[0], [2], [4], [6]]))
    sys.stdout.write("\r")

    print("%f %f %f %f" % (nn(np.array([0])), nn(np.array([1])), nn(np.array([2])), nn(np.array([3]))))

    nns = list(NeuralNetwork([NeuronLayer(2, 4, LeakyReLU()),
                              NeuronLayer(4, 2, LeakyReLU()),
                              NeuronLayer(2, 1, Identity(), hidden=False)]) for i in range(4))

    max_iter = 10000
    for i in range(max_iter):
        sys.stdout.write("\rProgress: %0.1f%%" % float(i/max_iter*100))
        for nn in nns:
            nn.learn(np.array([[0, 1], [1, 0], [1, 1], [0, 0]]), np.array([[0], [0], [1], [1]]), lr=10**-3)
    sys.stdout.write("\r")

    for nn in nns:
        print("%f %f %f %f" % (nn(np.array([0, 1])), nn(np.array([1, 0])), nn(np.array([1, 1])), nn(np.array([0, 0]))))


