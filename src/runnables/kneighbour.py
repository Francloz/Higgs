import os

from src.functions.distance import L2
from src.model.classifier.clustering import *
from src.utils.data_manipulation import *

if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'filled_dataset.npy')
    data[:, 22] /= 3

    train, test = split(data, test_relative_size=.9)

    y = np.expand_dims(train[:, 1], axis=1)
    tx = train[:, 2:]

    running_loss = 0
    batch_size = 1000
    max_batches = 5
    for k in range(1, 200, 2):
        model = KNeighbourhoodCluster(x=tx, y=y, distance=L1(), k=k)
        running_loss = 0
        iter = int(min(test.shape[0]/batch_size+.5, max_batches))
        for batch_y, batch_x in batch_iter(test[:, 1], test[:, 2:], 50, 50):
            prediction = np.reshape(model(batch_x), (-1, 1))
            label = np.reshape(batch_y, (-1, 1))
            running_loss += np.sum(L2()(prediction, label))
        print("For k=%d error=%f" % (k, running_loss/(iter*batch_size)))
