from src.regression.nn import *
from src.utils.activation_functions import *
import sys
import os

if __name__ == "__main__":
    # Test training linear simple regression
    nn = NeuralNetwork([NeuronLayer(1, 1, LeakyReLU(), hidden=False)])
    max_iter = 1000
    print_step = 1000
    for i in range(max_iter):
        nn.learn(np.array([[0], [1], [2], [3]]), np.array([[0], [2], [4], [6]]))
        if i % print_step == print_step-1:
            sys.stdout.write("\rProgress: %d%%" % int((i+1) / max_iter * 100))
            sys.stdout.write(" with values: %0.2f %0.2f %0.2f %0.2f \n" % (nn(np.array([0])),
                                                                           nn(np.array([1])),
                                                                           nn(np.array([2])),
                                                                           nn(np.array([3]))))
        else:
            sys.stdout.write("\rProgress: %d%%" % int(i / max_iter * 100))
    sys.stdout.write("\r")

    # Test saving and reading from a file.
    print(nn.layers[0].get_weights())
    nn.save('./param.txt')
    nn.load('./param.txt')
    print(nn.layers[0].get_weights())

    # Test training XOR logic door
    max_iter = 10000
    nns = list(NeuralNetwork([NeuronLayer(2, 4, LeakyReLU()),
                              NeuronLayer(4, 2, LeakyReLU()),
                              NeuronLayer(2, 1, Identity(), hidden=False)]) for i in range(4))
    for i in range(max_iter):
        for nn in nns:
            nn.learn(np.array([[0, 1], [1, 0], [1, 1], [0, 0]]), np.array([[0], [0], [1], [1]]), lr=10**-2)
        if i % print_step == print_step-1:
            sys.stdout.write("\rProgress: %d%% with values: \n" % int((i+1) / max_iter * 100))
            for nn in nns:
                sys.stdout.write(" %0.2f %0.2f %0.2f %0.2f \n" % (nn(np.array([0, 1])),
                                                                  nn(np.array([1, 0])),
                                                                  nn(np.array([1, 1])),
                                                                  nn(np.array([0, 0]))))
        else:
            sys.stdout.write("\rProgress: %d%%" % int(i / max_iter * 100))
    sys.stdout.write("\r")

    # Test saving the neural network
    print(nns[0].layers[1].get_weights())
    nn.save('./param.txt')
    nn.load('./param.txt')
    print(nns[0].layers[1].get_weights())

    os.remove('./param.txt')



