from src.model.nn import *
from src.functions.activation_functions import *
from src.utils.data_manipulation import *
from src.preconditioning.normalization import *
from src.functions.loss import *
from src.functions.distance import *
import sys
import os


# def train_nn():
#     parallel_nn = 10
#     epochs = 100
#     training_iter = 100
#
#     path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '\\resources\\train.csv'
#     data = load_data(path)
#
#     for i in range(epochs):
#         data_tra, data_test = split(data)
#         y = np.expand_dims(data_tra[:, 1], axis=1)
#         tx = data_tra[:, 2:]
#
#         normalizer = GaussianNormalizer()
#         tx = normalizer(tx)
#         outlier_removal = GaussianOutlierRemoval()
#         clean_tx = outlier_removal(tx, max_prob=0.001)
#
#         nns = list(NeuralNetwork([NeuronLayer(30, 60, LeakyReLU()),
#                                   NeuronLayer(60, 60, LeakyReLU()),
#                                   NeuronLayer(60, 20, LeakyReLU()),
#                                   NeuronLayer(20, 1, Sigmoid(), hidden=False)]) for i in range(parallel_nn))
#         for i in range(len(nns)):
#             path = './param' + str(i) + '.txt'
#             parameter_file = Path(path)
#             if parameter_file.is_file():
#                 nns[i].load(path)
#
#         for i in range(max_iter):
#             sys.stdout.write("\rProgress: %0.2f%%" % float(i/max_iter*100))
#             for batch_y, batch_tx in batch_iter(y, tx, 32):
#                 for nn in nns:
#                     nn.learn(batch_tx, batch_y, lr=10**-3)
#         sys.stdout.write("\rTraining complete.\n")
#         for i in range(len(nns)):
#             nns[i].save('./param' + str(i) + '.txt')
#
#         path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '\\resources\\test.csv'
#         data = load_data(path)
#         y = np.expand_dims(data[:, 1], axis=1)
#         tx = data[:, 2:]
#         tx = normalizer(tx)

# Now test the data
# max_tests = int(tx.shape[0]/100)
# i = -1
# loss = MAE()
# for nn in nns:
#     i += 1
#     # path = os.path.dirname(os.path.abspath(__file__)) + '\\nn_param\\param' + str(i) + '.txt'
#     running_loss = 0
#     for row in range(max_tests):
#         sys.stdout.write("\rEvaluating NN " + str(i) + ": %0.2f%%" % float(i/max_tests*100))
#         prediction = nn(tx[row, :])
#         running_loss += euclidean_distance(np.where(prediction < .5, 0, 1), y[row, :])
#     sys.stdout.write("\rError of NN " + str(i) + " is: ")
#     print("%0.1f%%" % float(running_loss/max_tests * 100))


def train_sgd():
    epochs = 100
    training_iter = 100

    path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '\\resources\\train.csv'
    data = load_data(path)

    w = np.zeros((30, 1))
    path = './progress/sgd.txt'
    parameter_file = Path(path)
    if parameter_file.is_file():
        file_array = np.fromfile(path, sep=" ")
        w = np.reshape(file_array, (30, 1))

    for i in range(epochs):
        data_tra, data_test = split(data)

        normalizer = GaussianNormalizer()
        loss = LogCosh()
        tr_y = np.expand_dims(data_tra[:, 1], axis=1)
        tr_x = normalizer(data_tra[:, 2:])
        te_y = np.expand_dims(data_test[:, 1], axis=1)
        te_x = normalizer(data_test[:, 2:])
        # outlier_removal = GaussianOutlierRemoval()
        # cleaner = outlier_removal(tx, max_prob=0.001)

        for i in range(training_iter):
            sys.stdout.write("\rProgress: %0.2f%%" % float(i/training_iter*100))
            for batch_y, batch_tx in batch_iter(tr_y, tr_x, 500):
                w -= 10**-3*loss.gradient(batch_tx, batch_y, w)
        sys.stdout.write("\rTraining complete.")
        f = open(path, "w+")
        np.savetxt(f, w)
        f.close()

        # Now test the data
        max_tests = int(te_x.shape[0]/100)
        sys.stdout.write("\rEvaluating the error.")
        sys.stdout.write("\rError: %0.3f\n" % float(np.sum(euclidean_distance(np.where(np.dot(te_x, w) < .5, 0, 1), te_y))))
        # loss(te_x, te_y, w))


if __name__ == "__main__":
    train_sgd()



