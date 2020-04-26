import numpy as np

class Linear_Regression:
    #Initialize Linear Regression Class with alpha parameter
    def __init__(self,alpha):
        self.alpha = alpha
        self.w = np.zeros(8)
    
    #Fitting the weights vector to the input data   
    def fit(self,X,y):
        N,n = np.shape(X)
        X_t = X.transpose()

        A = np.matmul(X_t,X)
        B = np.linalg.inv(A+(self.alpha*self.alpha)*np.identity(n))

        C = np.matmul(B,X_t)
        self.w = np.matmul(C,y)
        return self
    
    #Makes prediction using the dot product between w and x being w[0] the bias weight used for the bias
    def pred(self, test):
       P=[]
       for i in range(len(test)):
           prediction = np.dot(test[i], self.w)
           P.append(rounding(prediction)) 
       predictions=np.array(P)
       return predictions

#Rounding function used
def rounding(p):
    n=0
    if p>0.5:
        n=1
    return n
    