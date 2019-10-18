from src.optimization.linear import *
from src.model.classifier.clustering import *
from src.functions.activation_functions import *
from src.preconditioning.normalization import MinMaxNormalizer
from src.functions.distance import L2
import os


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\'
    # data = load_dataset(path + 'train.csv')
    data = np.load(file=path + 'train.npy')
    train, test = split(data)

    y = np.expand_dims(train[:, 1], axis=1)
    tx = train[:, 2:]
    normalizer = MinMaxNormalizer()
    tx = normalizer(tx)
    model = KNeighbourhoodCluster(tx=tx, y=y, distance=L2(), k=10)
    running_loss = 0
    for i in range(100):
        prediction = model(normalizer(test[i, 2:]))
        label = np.reshape(np.array(test[i, 1]), (1, 1))
        print(L2()(prediction, label))
    print(running_loss)
