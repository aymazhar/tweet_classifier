import numpy as np
from scipy import stats
from scipy.spatial import distance

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X 
        self.y = y

    def predict(self, Xtest):
        X = self.X
        y = self.y
        n = X.shape[0]
        t = Xtest.shape[0]
        k = min(self.k, n)
        
        # Compute cosine_distance distances between X and Xtest
        cosDist = self.cosine_distance(X, Xtest)

        # yhat is a vector of size t with integer elements
        yhat = np.ones(t, dtype=np.uint8)
        
        for i in range(t):
            # sort the distances to other points
            inds = np.argsort(cosDist[:,i])

            # compute mode of k closest training pts
            yhat[i] = stats.mode(y[inds[:k]])[0][0]

        return yhat


    def cosine_distance(self,X1,X2):
        X2 = np.transpose(X2)
        dot_prod = np.dot(X1,X2)
        x_norm = np.sqrt(np.sum(np.transpose(X1)**2))
        x_test_norm = np.sqrt(np.sum(X2**2))
        return 1 - dot_prod/(x_norm*x_test_norm)
        
    
    
    
    