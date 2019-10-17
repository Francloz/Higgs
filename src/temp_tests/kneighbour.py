from src.optimization.linear import *
from src.model.classifier.clustering import *
from src.functions.activation_functions import *
from src.preconditioning.normalization import GaussianNormalizer
from src.functions.distance import L2
import os


if __name__ == "__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\train.csv'
    data = load_dataset(path)

    train, test = split(data)

    y = np.expand_dims(train[:, 1], axis=1)
    tx = train[:, 2:]
    normalizer = GaussianNormalizer()
    tx = normalizer(tx)
    model = KNeighbourhoodCluster(tx=tx, y=y, distance=L2(), k=2)

    prediction = model(normalizer(test[1, 2:]))
    label = np.expand_dims(test[1, 1], axis=1)
    L2()(prediction, label)
