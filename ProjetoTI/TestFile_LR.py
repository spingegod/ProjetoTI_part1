from pulsar_project.Linear_Regression import Linear_Regression as L_R
from pulsar_project import Sampling
from pulsar_project import Metrics
from pulsar_project.Cross_Validation import Cross_Validation as CV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
ds = pd.read_csv('pulsar_stars.csv').values
full_ds=ds[:,0:8]
full_labels=ds[:,8]

#Train/Test spliting (90% train and 10% test)
from sklearn.model_selection import train_test_split
train, test, labels_train, labels_test = train_test_split(full_ds, full_labels, test_size=0.1, random_state=0)

#Appending ones in order to support affine linear functions
A = np.ones(train.shape[0])
B  = np.array([A]).T
train = np.hstack((train,B))
#Oversampling procedure
V=Sampling.oversampling(train,labels_train)

#Train values and classes
X=V[0]
y=V[1]

#Test values and classes
X_test=test
y_test=labels_test

#Appending ones in order to support affine linear functions for test points
A = np.ones(X_test.shape[0])
B  = np.array([A]).T
X_test = np.hstack((X_test,B))

#Cross-Validation to find ideal Alpha hiperparameter
MSE=[]
for alpha in np.linspace(0,20,200):
    MSE.append(CV(X, y, 10, alpha))
    
#Plotting MSE vs Alpha    
xx=np.linspace(0,20,200)
plt.plot(xx,MSE)
plt.xlabel('Alpha')
plt.ylabel('Mean squared error')
plt.title('Mean Squared error as function of Alpha')
plt.show()

#Optimal Alpha is argmin(MSE)
alpha=np.argmin(MSE)*(0.1)

#Train and predict with Linear regression with optimal alpha
LR = L_R(alpha)
LR.fit(X,y)
y_pred=LR.pred(X_test)

#Performance Metrics
ACC=Metrics.check_accuracy(y_pred, y_test)
PREC=Metrics.precision(y_pred, y_test)
REC=Metrics.recall(y_pred,y_test)
F1=Metrics.F1score(PREC,REC)

#Printing the performace
print('ALPHA =',alpha)
print('Weights: ',LR.w[0:-1],'Bias: ', LR.w[-1])
print('Linear regression accuracy is: ', ACC)
print('Linear regression precision is: ', PREC)
print('Linear regression recall is: ', REC)
print('Linear regression F1 score is: ', F1)

