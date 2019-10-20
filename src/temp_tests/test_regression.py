import numpy as np
import os
import sys
sys.path.append('C:\\Users\\Abhi Kamboj\\ML_Higgs')

from src.optimization.linear import *
#import src.model.regression.linear_model

if __name__=="__main__":
    # path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\train.csv'
    # data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, converters={1: lambda x: 1 if b"b" in x else 0})
    # np.save("traindata",data)
    data = np.load("traindata.npy")
    print("just loaded")
    print("Data:",data.shape)
    y = np.expand_dims(data[:, 1], axis=1)
    print(y)
    print("y", y.shape)
    tx = data[:, 2:]
    print("tx",tx.shape)
    tx_bias = np.ones((tx.shape[0],tx.shape[1]+1))
    print("tx",tx_bias.shape)

    #seems to work better without the bias
    tx_bias[:,:-1] = tx

    m = LinearModel(tx.shape[0])
    r = Ridge()
    print(m.w)

    lambdas = np.logspace(-5, 0, 15)
    for lam in lambdas: 
        r(m,tx,y,lambdas[0])
        y_guess = np.array(list(map(lambda x: 1 if x>.5 else 0, m(tx))))
        num_correct = np.sum(np.logical_xor(y_guess,y.transpose()))
        print("With Lamda:",lam, "correct:",num_correct)
    




