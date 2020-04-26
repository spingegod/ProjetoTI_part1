from pulsar_project.MinOver import Perceptron_MinOver as ANN
from pulsar_project import Sampling
from pulsar_project import Metrics
import pandas as pd
import sklearn.preprocessing as skp

ds = pd.read_csv('pulsar_stars.csv').values
full_ds=ds[:,0:8]
full_ds_norm = skp.normalize(full_ds,axis=0)
full_labels=ds[:,8]
#Imports the dataset

from sklearn.model_selection import train_test_split
train_norm, X_test, labels_train_norm, y_test = train_test_split(full_ds_norm, full_labels, test_size=0.05, random_state=0)
#Splits the dataset into training(95%) and test(5%)


V=Sampling.ru_sampling(train_norm,labels_train_norm)#UnderSampling applied in the training data

X=V[0]
y=V[1]

Per=ANN(1000)#Creates an object of the class Perceptron_MinOver with the maximum number of epochs=1000
Per.fit(X,y)#Iterates through feature vectors and trains the model
y_pred=Per.pred(X_test)#Predicts the test labels
ACC=Metrics.check_accuracy(y_pred, y_test)#Computes accuracy
PREC=Metrics.precision(y_pred, y_test)#Precision
REC=Metrics.recall(y_pred,y_test)#Recall
F1=Metrics.F1score(PREC,REC)#F1 Score

print('MinOver Perceptron accuracy is: ', ACC)
print('MinOver Perceptron precision is: ', PREC)
print('MinOver Perceptron recall is: ', REC)
print('MinOver Perceptron F1 score is: ', F1)
