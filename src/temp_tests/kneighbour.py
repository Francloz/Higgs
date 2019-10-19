from src.optimization.linear import *
from src.model.classifier.clustering import *
from src.functions.activation_functions import *
from src.preconditioning.normalization import *
from src.functions.distance import L2
from src.utils.data_manipulation import *
import os


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    data = np.load(file=path + 'train.npy')
    train, test = split(data)

    y = np.expand_dims(train[:, 1], axis=1)
    tx = train[:, 2:]
    normalizer = GaussianNormalizer()
    tx = normalizer(tx)
    running_loss = 0
    batch_size = 100
    max_batches = 5
    for k in range(10, 200, 5):
        model = KNeighbourhoodCluster(x=tx, y=y, distance=L1(), k=k)
        running_loss = 0
        iter = int(min(test.shape[0]/batch_size+.5, max_batches))
        for i in range(iter):
            prediction = np.reshape(model(normalizer(test[i*batch_size:(i+1)*batch_size, 2:])), (-1, 1))
            label = np.reshape(np.array(test[i*batch_size:(i+1)*batch_size, 1]), (-1, 1))
            running_loss += np.sum(L2()(prediction, label))
        print("For k=%d error=%f" % (k, running_loss/(iter*batch_size)))
