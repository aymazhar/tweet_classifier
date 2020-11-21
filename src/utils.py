import numpy as np
from scipy import stats



"""Given a numpy array, returns element with max count"""
def mode(y):
    if len(y)==0:
        return -1
    return stats.mode(y.flatten())[0][0]


"""Given X (N x D numpy array), and Xtest (T x D numpy array), 
Computes the Euclidean distance between rows of X and rows of Xtest
Return value: N x T array with pairwise squared Euclidean distance """ 
def euclidean_dist_squared(X, Xtest):
    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)


def evaluate_model(model,X,y,X_test,y_test):
    model.fit(X,y)
  
    y_pred = model.predict(X)
    tr_error = np.mean(y_pred != y)
    
    y_pred = model.predict(X_test)
    te_error = np.mean(y_pred != y_test)
    
    print("    Training error: %.3f" % tr_error)
    print("    Testing error: %.3f" % te_error)