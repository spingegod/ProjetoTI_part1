import numpy as np
import math

class Perceptron_MinOver:# Code to generate objects from the Perceptron class using MinOver algortihm for stability

    
  def __init__(self, ns):# Constructor
    self.ns = ns# Number of epochs
    

  def fit(self, X, y):# Training the Perceptron
    n_f = X.shape[1]
    self.w = np.zeros(n_f)# Initialization of weights vector with zeros
    e=0
    E=[]
    self.A=[]
    for i in range(self.ns):# Starting the iterations through epochs
      e=e+1
      K=[]
      L=[]
      for s, y_true in zip(X, y):  
        k=np.dot(self.w,s)*y_true/np.linalg.norm(self.w)#Computes stability
        K.append(k)
        y_pred    = self.pred(s)
        d = (y_true - y_pred)
        L.append(d)
        acc=self.accuracy(L)*100
      #print('Epoch: ' + str(e) + ' ; ' + 'Fitting accuracy: ' + str(round(acc,2)) + ' %')
      i_min = min(range(len(K)), key=K.__getitem__)#Checks index of feature vector with minimal stability
      ac = ((1/np.pi)*math.acos(np.dot(self.w,self.w + (1/n_f)*X[i_min]*y[i_min])/(np.linalg.norm(self.w)*np.linalg.norm(self.w + (1/n_f)*X[i_min]*y[i_min]))))
      if ac<0.005:#Here optimal stability is considered if the angular change between updates is smaller than 0.005 ()
          print('Optimal Stability was acheived in '+str(e)+' epochs')
          break
      self.w=self.w + (1/n_f)*X[i_min]*y[i_min]
      self.A.append(acc)
      E.append(e)
    if i==self.ns-1:
        print("The maximum number of epochs was reached")  
    return self
      
      
  def accuracy(self, D):# Computes accuracy in a given epoch
      N=[]
      for i in range(len(D)):
          if D[i]==0:
              N.append(1)
      acc=len(N)/len(D)
      return acc
      
      
  def pred(self, s):# Makes prediction using the dot product between w and x
    prediction = np.dot(s, self.w)
    return np.where(prediction > 0, 1, -1)
