import numpy as np

class Perceptron:# Code to generate objects from the Perceptron class

    
  def __init__(self, ns):# Constructor
    self.ns = ns# Number of epochs
    

  def fit(self, X, y):# Training the Perceptron
    n_f = X.shape[1]
    self.w = np.zeros(n_f)# Initialization of weights vector with zeros 
    e=0
    E=[]
    A=[]
    for i in range(self.ns):# Starting the iterations through epochs
      e=e+1
      L=[]
      for s, y_true in zip(X, y):
        y_pred    = self.pred(s)# Make prediction
        d    = (y_true - y_pred)# Integer that defines if a label was predicted correctly or not (d=0 if it is a correct prediction or d=+-1 if its not)
        if (d == 2)or(d == -2):
            d==1
        w_new = d  # Compute weight update via Perceptron Learning Rule (equivalent to product of the dot product between sample and weight vector and true label)
        self.w    += (1/n_f)*w_new * s
        L.append(d)
        acc=self.accuracy(L)*100# Compute accuracy 
      A.append(acc)
      E.append(e)
      if acc==100:
          break# Breaking loop if 100% accuracy is reached 
    return self  
      
  def accuracy(self, D):# Computes accuracy in a given epoch
      N=[]
      for i in range(len(D)):
          if D[i]==0:
              N.append(1)#N counts the amount of labels guessed right
      acc=len(N)/len(D)
      return acc
    
  def pred(self, s):# Makes prediction using the dot product between w and x
    prediction = np.dot(s, self.w)
    return np.where(prediction > 0, 1, 0)
