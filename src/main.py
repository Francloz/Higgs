import numpy as np
from src.regression.ridge import ridge_regression
from src.model.regression.linear_model import LinearModel
import os
from src.utils.data_manipulation import load_dataset
from src.model.classifier.logistic import Logistic
from src.preconditioning.normalization import MinMaxNormalizer
from src.preconditioning.feature_filling import MeanFilling
from src.model.nn.nn import NeuralNetwork, NeuronLayer
from src.functions.activation_functions import *
from src.utils.data_manipulation import *
import sys

if __name__ == "__main__":
    path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
    data = np.load(file=path + '\\resources\\' + 'train.npy')
    #
    # # np.save(arr=, file='./ridge.npy')
    # model = LinearModel((31, 1))
    # # array = np.load('./ridge.npy')
    # model.set_param(ridge_regression(training[:, 1], np.hstack([training[:, 2:]])))
    #
    normalizer = MinMaxNormalizer()
    filler = MeanFilling(data[:, 2:])
    data[:, 2:] = normalizer(filler(data[:, 2:]))
    # prediction = np.reshape(np.where(model(input) >= .5, 1, 0), (-1, 1))
    # indexes = np.reshape(data[:, 0], (-1, 1))
    #
    # x = np.hstack([indexes, prediction]).astype(int)
    # np.savetxt(X=x,
    #            fmt='%d',
    #            fname='./prediction.txt',
    #            delimiter=',',
    #            newline='\n')
    #
    # np.save(arr=ridge_regression(data[:, 1], np.hstack([data[:, 2:]])), file='./ridge.npy')
    #
    # features = np.hstack([data[:, 2:][:, [4, 5, 12, 27, 28, 0]],
    #                       data[:, 2:][:, [4, 5, 12, 27, 28, 0]] ** 2])
    # model = Logistic(features.shape[1])
    # kwargs = [{'batch_size': 25, 'lr': 10**-3, 'epochs': 1000, 'regularize': r}
    #           for r in np.logspace(0, 2, 10)]


    parallel_nn = 10
    max_iter = 100
    mask = [False, False, True, True,  True, False,  True,  True,  True,  True, False,  False,  True,  True,
            True,  True,  True,  True,  True,  True,  True, False,  True,  True,  True, False,
            True,  True,  True,  True,  True,  True]
    nns = list(NeuralNetwork([NeuronLayer(np.sum(mask), 60, LeakyReLU()),
                              NeuronLayer(60, 60, LeakyReLU()),
                              NeuronLayer(60, 20, LeakyReLU()),
                              NeuronLayer(20, 1, Sigmoid(), hidden=False)]) for i in range(parallel_nn))
    np.random.seed(0)
    min_error = np.inf
    iter_error = np.inf
    iter_best = None
    for epoch in range(1000):
        for i in range(max_iter):
            sys.stdout.write("\rProgress: %0.2f%%" % float(i/max_iter*100))
            training, test = split(data)
            for batch_y, batch_tx in batch_iter(np.reshape(training[:, 1], (-1, 1)), training[:, mask], batch_size=1,
                                                num_batches=1000):
                for nn in nns:
                    iter_error = np.inf
                    nn.learn(batch_tx, batch_y, lr=10**-1 * np.power(0.9, epoch))

            for nn in nns:
                iter_error = np.inf
                prediction = np.where(nns[i](test[:, mask]) > .5, 1, 0)
                correct_label = np.reshape(test[:, 1], (-1, 1))
                error = np.sum(np.abs(prediction - correct_label))/prediction.size

                if error < iter_error:
                    iter_error = error
                    iter_best = nns[i]

                if iter_error < min_error:
                    min_error = iter_error
                    best = iter_best
                    best.save('./nn' + str("%0.2f" % min_error) + ".npy")

            # plot(lambdas, np.array(errors), label=('Lambda', 'Error', str(normalizer) + ' and ' + str(filler))).savefig(str(filler) + str(normalizer) + '.png')
            print("Error for model", i, "using ", normalizer, filler, " is ", min_error)

    sys.stdout.write("\rTraining complete.\n")
    for i in range(len(nns)):
        nns[i].save('./param' + str(i) + '.txt')

    path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '\\resources\\temp_tests.csv'
    data = load_dataset(path)
    y = np.expand_dims(data[:, 1], axis=1)
    tx = data[:, 2:]
    tx = normalizer(tx)

    # Now temp_tests the data
    max_tests = int(tx.shape[0]/100)
    i = -1
    loss = MAE()
    for nn in nns:
        i += 1
        # path = os.path.dirname(os.path.abspath(__file__)) + '\\nn_param\\param' + str(i) + '.txt'
        running_loss = 0
        for row in range(max_tests):
            sys.stdout.write("\rEvaluating NN " + str(i) + ": %0.2f%%" % float(i/max_tests*100))
            prediction = nn(tx[row, :])
            running_loss += L2()(np.where(prediction < .5, 0, 1), y[row, :])
        sys.stdout.write("\rError of NN " + str(i) + " is: ")
        print("%0.1f%%" % float(running_loss/max_tests * 100))

