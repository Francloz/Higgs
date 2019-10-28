"""
This file runs the script for our final submission on AIcrowd.
"""

import csv
import os
import sys
sys.path.append(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]) #'C:\\Users\\Abhi Kamboj\\ML_Higgs')

from src.optimization.linear import *
from src.preconditioning.feature_elimination import*

def process_avg_xs(x):
    """
    inputs x: data points as rows and features as columns
    output: resulting processed data points
    """
    avg = x.mean(0) #average across 0th axis

    for ind,feat in enumerate(x.transpose()):
        #feat is also a reference to row
        feat[feat==-999] = avg[ind]
        feat[feat==0] = avg[ind]

    return x

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w', newline='') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


if __name__ == "__main__":
    
    path = os.path.dirname(os.path.abspath(__file__)) + '\\resources\\train.csv'
    # print(os.path.dirname(os.path.abspath(__file__)))
    pathtest= os.path.dirname(os.path.abspath(__file__)) + '\\resources\\test.csv'
    
    print("Loading train.csv")
    data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, converters={1: lambda x: 1 if b"b" in x else 0})
    
    print("Loading test.csv")
    ys, x_test, ids = load_csv_data(pathtest)  # ignore ys, they're just question marks
    ids = ids.reshape((-1, 1))
    # print(ys.shape,xs.shape,ids.shape,ys)

    # data = np.load("traindata.npy")
    print("Data:", data.shape)

    # extract x and ys
    y = np.expand_dims(data[:, 1], axis=1)
    print(y)
    print("y", y.shape)

    # Combine data for preconditionaing: tx = x_train and x_test
    tx = np.vstack((data[:, 2:], x_test))
    tx = np.delete(tx, [11], 1)  # remove 11
    print("tx shape (should be 818238x29): ", tx.shape)

    # fill in -999 and 0s
    tx = process_avg_xs(tx)

    # choose a normalizer
    MMN = MinMaxNormalizer()
    tx = MMN(tx)

    # add bias term
    tx = np.hstack([tx, np.ones(tx.shape[0]).reshape(-1, 1)])
    print("with bias", tx.shape)

    # split data again: different files (ACTUAL TESTING FOR SUBMISSION)
    x_train = tx[:250000, :]
    x_test = tx[250000:, :]
    # print("train size should be like 250000",train_size[0])

    # get the model and regression
    m = LinearModel(x_train.shape[1]) 
    r = Ridge()

    # .5 seems to be performing the best
    lam = .5
    r(m, x_train, y, lambda_=lam)

    # predict based on the training data
    y_guess = np.array(list(map(lambda x: 1 if x > .5 else 0, m(x_train))))
    # print("sum of guess: ",np.sum(y_guess))
    num_correct = len(y_guess) - np.sum(np.logical_xor(y_guess, y.transpose()))
    # print("With Lamda:",lam, "correct:",num_correct)

    print("Accuracy on TrainingSet: ", num_correct / 250000, "lambda: ", lam)

    # now test it with the reallll data...
    print("testing ids", ids.shape, ids)
    print("testing data", x_test.shape, x_test)

    y_predict = np.array(list(map(lambda x: 1 if x <= .5 else -1, m(x_test))))
    print("y predict", y_predict.shape, y_predict)

    create_csv_submission(ids, y_predict, "pred_ridge_uploaded.csv")
