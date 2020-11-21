import argparse
import numpy as np
import pandas as pd                             


import utils
from knn import KNN
from naive_bayes import NaiveBayes
from random_forest import RandomForest
from stacking import Stacking

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--classifier', required=True)

    io_args = parser.parse_args()
    classifier = io_args.classifier
    
    #init datasets
    df_train = pd.read_csv('../data/wv_train.csv')
    df_test = pd.read_csv('../data/wv_test.csv')
                    
    X = np.array(df_train)[:,:-1] 
    y = np.array(df_train)[:,-1]
    
    Xtest = np.array(df_test)[:,:-1]
    ytest = np.array(df_test)[:,-1]
    
  
    if classifier == 'rf':   #RandomForests
        print("Running Random Forests")
        model = RandomForest(num_trees=15, max_depth=np.inf)
        utils.evaluate_model(model, X, y.flatten(), Xtest, ytest.flatten())


    elif classifier == 'nb':  #NaiveBayes
        print("Running NaiveBayes")
        model = NaiveBayes()
        utils.evaluate_model(model, X, y.flatten(), Xtest, ytest.flatten())


    elif classifier == 'knn': #KNearestNeighbours
        print("Running KNN")              
        model = KNN(k=3)
        utils.evaluate_model(model, X, y.flatten(), Xtest, ytest.flatten())


    elif classifier == 'stack': #Stacking RF, NB and KNN. Metaclassifier = DT
        print("Running Stacking Classifier")
        model = Stacking()
        utils.evaluate_model(model, X, y.flatten(), Xtest, ytest.flatten())
    
    else:
        print("Unknown classifier: %s" % classifier)
