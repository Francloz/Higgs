import numpy as np
import os
import sys
sys.path.append(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]) #'C:\\Users\\Abhi Kamboj\\ML_Higgs')

from src.optimization.linear import *
from src.preconditioning.normalization import*
from src.preconditioning.feature_elimination import*
#import src.model.regression.linear_model

def process_avg_xs(x):
    """
    Note: Np arrays are pass by reference--
        changing the value here will change the actual value!
    inputs x: data points as rows and features as columns
    """
    avg = x.mean(0) #average across 0th axis
    print(avg)

    for ind,feat in enumerate(x.transpose()):
        #feat is also a reference to row
        feat[feat==-999] = avg[ind]

    return x

def processx_remData(x,y):
    """ delete rows where value is -999 
    """
    to_del = np.where(x==-999)[0]
    x = np.delete(x,to_del,0)
    y = np.delete(y,to_del,0)

    return x,y

def processx_remFeat(x):
    print(x.shape)
    x = np.delete(x,np.where(x==-999)[1],1)
    print(x.shape)
    return x  

if __name__=="__main__":
    # path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\train.csv'
    # data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, converters={1: lambda x: 1 if b"b" in x else 0})
    # np.save("traindata",data)
    data = np.load("traindata.npy")
    print("just loaded")
    print("Data:",data.shape)
    g = GaussianNormalizer()
    MMN = MinMaxNormalizer()
    GOR = GaussianOutlierRemoval()

    y = np.expand_dims(data[:, 1], axis=1)
    print(y)
    print("y", y.shape)
    tx = data[:, 2:]
    
    v_featElim = VarianceThreshold(tx)
    tx = v_featElim(tx)
    print("x shape: ",tx.shape)
   # tx,y = processx_remData(tx,y)
   # tx = process_avg_xs(tx)
   # tx = process_avg_xs(tx)

    #tx = g(tx)
    tx=MMN(tx)
    #tx = GOR(tx)

    tx_train,tx_test = split(tx)
    y_train,y_test = split(y)
    m = LinearModel(tx.shape[0])
    r = Ridge()
    print(m.w)

    
    #r(m,tx,y,.2)
    highm =0
    highlam = 0
    lambdas = np.logspace(-5,0,15) #np.linspace(-100,100,5001)
    
    for ind,lam in enumerate(lambdas): 
        r(m,tx_train,y_train,lam)
        y_guess = np.array(list(map(lambda x: 1 if x>.5 else 0, m(tx_test))))
        #print("sum of guess: ",np.sum(y_guess))
        num_correct = len(y_guess) - np.sum(np.logical_xor(y_guess,y_test.transpose()))
        #print("With Lamda:",lam, "correct:",num_correct)
        if num_correct>highm:
            highm=num_correct
            highlam = lam        
        if ind%500==0:
            print("On: ",ind)
    print("Max: ", highm/(250000*.3), highm, "lam: ",highlam)

    l = LS()
    l(m,tx_train,y_train)
    y_guess = np.array(list(map(lambda x: 1 if x>.5 else 0, m(tx_test))))
        #print("sum of guess: ",np.sum(y_guess))
    num_correct = len(y_guess) - np.sum(np.logical_xor(y_guess,y_test.transpose()))

    print("LeastSquares: ",num_correct/(250000*.3))

    




