import numpy as np
import csv
import os
import sys
sys.path.append(os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0]) #'C:\\Users\\Abhi Kamboj\\ML_Higgs')

from src.optimization.linear import *
from src.preconditioning.normalization import*
from src.preconditioning.feature_elimination import*

def process_avg_xs(x):
    """
    Note: Np arrays are pass by reference--
        changing the value here will change the actual value!
    inputs x: data points as rows and features as columns
    """
    avg = x.mean(0) #average across 0th axis
    #print(avg)

    for ind,feat in enumerate(x.transpose()):
        #feat is also a reference to row
        feat[feat==-999] = avg[ind]
        feat[feat==0] = avg[ind]

    return x

def processx_remData(x,y):
    """ delete rows where value is -999 
    """
    to_del = np.where(x==-999)[0]
    x = np.delete(x,to_del,0)
    y = np.delete(y,to_del,0)

    return x,y

def processx_remFeat(x):
    """removes features that contains a -999"""
    print(x.shape)
    x = np.delete(x,np.where(x==-999)[1],1)
    print(x.shape)
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


if __name__=="__main__":
    path = os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\train.csv'
    pathtest= os.path.split(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])[0] + '\\resources\\test.csv'
    # data = np.loadtxt(open(path, "rb"), delimiter=",", skiprows=1, converters={1: lambda x: 1 if b"b" in x else 0})
    # np.save("traindata",data)
    
    # ys,xs,ids = load_csv_data(pathtest)
    # ids=ids.reshape((-1,1))
    # print(ys.shape,xs.shape,ids.shape,ys)
    # np.save("testdata",np.hstack([ids,xs]))
    
    #load test data
    testdata = np.load("testdata.npy")
    x_test = testdata[:,1:]
    ids = testdata[:,0]

    data = np.load("traindata.npy")
    print("Data:",data.shape)

    #extract x and ys
    y = np.expand_dims(data[:, 1], axis=1)
    print(y)
    print("y", y.shape)
    x_train = data[:, 2:]
    train_size = x_train.shape
    #tx is TOTAL x with x_train and x_test
    tx = np.vstack((x_train,x_test))
    tx = np.delete(tx,[5,11,25],1) #remove 5 11 25
                                        #remove feature 3,8,19,23
                                        #remove 11


    #fill in -999 and 0s
    tx = process_avg_xs(tx)
   # tx,y = processx_remData(tx,y)
   # tx = process_avg_xs(tx)

    #feature elimination
    #v_featElim = VarianceThreshold(tx)
    #tx = v_featElim(tx)
    #250,000 test data points and 568,239 train data points
    print("tx shape (should be 818238x31): ",tx.shape)


    #choose a normalizer
    g = GaussianNormalizer()
    MMN = MinMaxNormalizer()
    # tx = g(tx)
    tx=MMN(tx)

    #add bias term
    tx = np.hstack([tx,np.ones(tx.shape[0]).reshape(-1,1)])
    print("with bias",tx.shape)

    # #split the data: same file
    # train,test = split(np.hstack([y,tx]))
    # y_train,y_test = train[:,0],test[:,0]
    # tx_train,tx_test = train[:,1:],test[:,1:]
    # print(y_train.shape,tx_train.shape)

    #split data: different files (ACTUAL TESTING FOR SUBMISSION)
    x_train = tx[:train_size[0],:]
    x_test = tx[train_size[0]:,:]
    print("trainsize should be like 250000",train_size[0])

    #get the model and regression
    m = LinearModel(tx.shape[1]) #doesn't matter for ridge bc ridge set's parameters not update existing
    r = Ridge()

    
    #keep track of highest value
    highm =0
    highlam = 0

    lambdas = [.5] #np.logspace(-5,0,50) #np.linspace(-100,100,5001)
    #.5 seems to be performing the best
    for ind,lam in enumerate(lambdas): 
        r(m,x_train,y,lambda_=lam)

        #predict based on the training data
        y_guess = np.array(list(map(lambda x: 1 if x>.5 else 0, m(x_train))))
        #print("sum of guess: ",np.sum(y_guess))
        num_correct = len(y_guess) - np.sum(np.logical_xor(y_guess,y.transpose()))
        #print("With Lamda:",lam, "correct:",num_correct)
        if num_correct>highm:
            highm=num_correct
            highlam = lam        
        if ind%500==0:
            print("On: ",ind)

    print("Max: ", highm/(250000), highm, "lam: ",highlam)

    #now test it with the reallll data...
    print("testing ids",ids.shape,ids)
    print("testing data",x_test.shape,x_test)

    y_predict= np.array(list(map(lambda x: 1 if x<=.5 else -1, m(x_test))))
    print("y predict",y_predict.shape,y_predict)

    create_csv_submission(ids,y_predict,"pred_ridge_sans51125.csv")


    # l = LS()
    # l(m,tx,y)
    # y_guess = np.array(list(map(lambda x: 1 if x>.5 else 0, m(tx))))
    #     #print("sum of guess: ",np.sum(y_guess))
    # num_correct = len(y_guess) - np.sum(np.logical_xor(y_guess,y.transpose()))

    # print("LeastSquares: ",num_correct/(250000))

    




