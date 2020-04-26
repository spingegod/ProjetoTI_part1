import pandas as pd
from pulsar_project import Metrics
from pulsar_project import Sampling
from sklearn.ensemble import RandomForestClassifier as RF

#Importing the dataset
ds = pd.read_csv('pulsar_stars.csv').values
full_ds=ds[:,0:8]
full_labels=ds[:,8]

#Train/Test spliting (90% train and 10% test)
from sklearn.model_selection import train_test_split
train, test, labels_train, labels_test = train_test_split(full_ds, full_labels, test_size=0.1, random_state=0)

#Undersampling procedure
V=Sampling.undersampling(train,labels_train)

#Train values and classes
X=V[0]
y=V[1]

#Test values and classes
X_test=test
y_test=labels_test

#Training Decision Tree
clf=RF(1000)
clf.fit(train,labels_train)

#Predicting test labels
y_pred=clf.predict(test)

#Performance Metrics
ACC=Metrics.check_accuracy(y_pred, y_test)
PREC=Metrics.precision(y_pred, y_test)
REC=Metrics.recall(y_pred,y_test)
F1=Metrics.F1score(PREC,REC)

print('Random Forest classifier accuracy is: ', ACC)
print('Random Forest classifier precision is: ', PREC)
print('Random Forest classifier recall is: ', REC)
print('Random Forest classifier F1 score is: ', F1)
