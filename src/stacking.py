import numpy as np
from random_forest import RandomForest, DecisionTree, DecisionStumpErrorRate
from knn import KNN
from naive_bayes import NaiveBayes


class Stacking():

    def __init__(self):
        pass

    def fit(self, X, y):
        self.rf = RandomForest(num_trees=15, max_depth=np.inf)
        self.rf.fit(X,y)
        y_rf = self.rf.predict(X)
        
        self.nb = NaiveBayes()
        self.nb.fit(X,y)
        y_nb = self.nb.predict(X)
        
        self.knn = KNN(k=3)
        self.knn.fit(X,y)
        y_knn = self.knn.predict(X)
        
        newX = np.array([y_rf, y_nb, y_knn]).transpose()
        
        model = DecisionTree(max_depth=np.inf, stump_class=DecisionStumpErrorRate)
        self.model = model          
        
        model.fit(newX, y)


    def predict(self, X):
        y_rf = self.rf.predict(X)
        y_nb = self.nb.predict(X)
        y_knn = self.knn.predict(X)
        x_test = np.array([y_rf, y_nb, y_knn]).transpose()
        
        return self.model.predict(x_test)
    