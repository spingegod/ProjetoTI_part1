import numpy as np
from pulsar_project.Metrics import check_accuracy as check_accuracy
from pulsar_project.Linear_Regression import Linear_Regression as Linear_Regression

#Simple cross validation function for the linear regression
def Cross_Validation(X,y,k,a):
#X is the train dataset
#y is the labels vector
#a is the alpha value for the regression
#k is the number of folds
    n_s=np.shape(X)[0]
    k_size=round(n_s/k)
    X_Tests=[]
    X_Trains=[]
    Y_Tests=[]
    Y_Trains=[]
    S=[]
    A=[]
    for i in range(0,k_size*(k),k_size):
        S.append(i)
    S.append(n_s-1)
    
    for i in range(1,len(S)):
        #Returns the indices of the vectors to delete in order to obtain the train set:
        train_del=np.arange(S[i-1],S[i])
        #Returns the indices of the vectors to delete from the original dataset in order to btain the test set:
        test_del1=np.arange(0,S[i-1])
        test_del2=np.arange(S[i],S[-1]+1)
        test_del=np.concatenate((test_del1,test_del2),axis=None)
        X1_test=np.delete(X,test_del,0)#creates the ith test dataset
        X1_train=np.delete(X,train_del,0)#creates the ith train dataset
        X_Tests.append(X1_test)#list with all the test datasets 
        X_Trains.append(X1_train)#list with all the training datasets
        y1_train=np.delete(y,train_del,0)
        y1_test=np.delete(y,test_del,0)
        Y_Trains.append(y1_train)#List with all the train labels
        Y_Tests.append(y1_test)#List with all the test labels
    for i in range(len(X_Tests)):#Iterates through aall the different datasets and fits a linear regression to each of them 
        LR=Linear_Regression(a)
        LR.fit(X_Trains[i],Y_Trains[i])
        y_pred=LR.pred(X_Tests[i])
        acc=check_accuracy(y_pred,Y_Tests[i])
        A.append(acc)#List with all accuracies from all the combinations
    
    return A
