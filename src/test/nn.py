from src.regression.nn import *
from src.utils.activation_functions import *

if __name__ == "__main__":

    nn = NeuralNetwork([NeuronLayer(1, 1, Identity(), hidden=False)])
    for i in range(50):
        x1, x2, x3, x4 = np.array([0]), \
                         np.array([1]), \
                         np.array([2]), \
                         np.array([3])
        y1 = nn(x1)
        nn.back_prop(np.array([0]))
        y2 = nn(x2)
        nn.back_prop(np.array([2]))
        y3 = nn(x3)
        nn.back_prop(np.array([4]))
        y4 = nn(x4)
        nn.back_prop(np.array([6]))

    print("%f %f %f %f" % (y1, y2, y3, y4))

    nn = NeuralNetwork([NeuronLayer(2, 3, Sigmoid()),
                        NeuronLayer(3, 2, Sigmoid()),
                        NeuronLayer(2, 1, Identity(), hidden=False)])

    for i in range(1, 10000):

        x1, x2, x3, x4 = np.array([0, 1]),\
                         np.array([1, 0]),\
                         np.array([1, 1]),\
                         np.array([0, 0])
        y1 = nn(x1)
        nn.back_prop(np.array([0]))
        y2 = nn(x2)
        nn.back_prop(np.array([0]))
        y3 = nn(x3)
        nn.back_prop(np.array([1]))
        y4 = nn(x4)
        nn.back_prop(np.array([1]))

    print("%f %f %f %f" % (y1, y2, y3, y4))


