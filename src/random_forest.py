import numpy as np
import utils
from kmeans import Kmeans
from scipy import stats


class DecisionStumpErrorRate:

    def __init__(self):
        pass

    def fit(self, X, y, thresholds=None):
        N, D = X.shape

        # Get an array with the number of 0's, number of 1's, etc.
        count = np.bincount(y.astype(int))

        # Get the index of the largest value in count.
        y_mode = np.argmax(count)

        self.splitSat = y_mode
        self.splitNot = None
        self.splitVariable = None
        self.splitValue = None

        # If all the labels are the same, no need to split further
        if np.unique(y).size <= 1:
            return

        minError = np.sum(y != y_mode)

        # Loop over features looking for the best split
        for d in range(D):
            for n in range(N):
                # Choose value to equate to
                value = X[n, d]
                # Find most likely class for each split
                y_sat = utils.mode(y[X[:, d] > value])
                y_not = utils.mode(y[X[:, d] <= value])

                # Make predictions
                y_pred = y_sat * np.ones(N)
                y_pred[X[:, d] <= value] = y_not

                # Compute error
                errors = np.sum(y_pred != y)

                # Compare to minimum error so far
                if errors < minError:
                    # Store lowest error
                    minError = errors
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = y_sat
                    self.splitNot = y_not
        
        
    def predict(self, X):
        splitVariable = self.splitVariable
        splitValue = self.splitValue
        splitSat = self.splitSat
        splitNot = self.splitNot

        M, D = X.shape

        if splitVariable is None:
            return splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, splitVariable] > splitValue:
                yhat[m] = splitSat
            else:
                yhat[m] = splitNot

        return yhat


"""
A helper function that computes the Gini_impurity of the
discrete distribution p.
    """
def Gini_impurity(p):
        gi = 0*p
        for i in range(len(p)):
            gi[i] = p[i]*(1-p[i])
        return np.sum(gi)


class DecisionStumpGiniIndex(DecisionStumpErrorRate):

    def fit(self, X, y, split_features, thresholds):
        
        N, D = X.shape
        count = np.bincount(y.astype(int))
        p = count/np.sum(count) #convert counts to prbs
        imp_before_split = Gini_impurity(p)
        
        maxGain = 0
        self.splitVariable = None
        self.splitValue = None
        self.splitSat = np.argmax(count)
        self.splitNot = None
        
        if np.unique(y).size <= 1:
            return
        
        if split_features is None:
            split_features = range(D)
        
        #for each of the selected features, get the unique thresholds
        #find the min gini index for these values
        for d in split_features:
            threshs = np.unique(thresholds[d])
            for value in threshs:
                
                y_vals = y[X[:,d] > value]
                countL = np.bincount(y_vals.astype(int), minlength=len(count))
                countR = count - countL
                #compute gini impurity
                pL = countL/np.sum(countL)
                pR = countR/np.sum(countR)
                g_impL = Gini_impurity(pL)
                g_impR = Gini_impurity(pR)
                probL = np.sum(X[:,d] > value)/N
                probR = 1-probL
                g_index = np.sum(probL*g_impL) + np.sum(probR*g_impR)
                
                #represents the largest difference between the impurity before
                giniGain = imp_before_split - g_index
                
                if (giniGain > maxGain):
                    maxGain = giniGain
                    self.splitVariable = d
                    self.splitValue = value
                    self.splitSat = np.argmax(countL)
                    self.splitNot = np.argmax(countR)
            
    




"""**Decision Tree**"""

class DecisionTree:

    def __init__(self, max_depth, stump_class=DecisionStumpErrorRate):
        self.max_depth = max_depth
        self.stump_class = stump_class

    def fit(self, X, y, thresholds=None):
        # Fits a decision tree via greedy recursive splitting
        N, D = X.shape

        # Learn a decision stump
        splitModel = self.stump_class()
        splitModel.fit(X, y, thresholds=thresholds)

        if self.max_depth <= 1 or splitModel.splitVariable is None:
            # If we reach max depth or the decision stump does
            # nothing, use the decision stump

            self.splitModel = splitModel
            self.subModel1 = None
            self.subModel0 = None
            return

        # Fit a decision tree to each split, decreasing maximum depth by 1
        j = splitModel.splitVariable
        value = splitModel.splitValue

        # Find indices of examples in each split
        splitIndex1 = X[:, j] > value
        splitIndex0 = X[:, j] <= value

        # Fit decision tree to each split
        self.splitModel = splitModel
        self.subModel1 = DecisionTree(self.max_depth - 1, stump_class=self.stump_class)
        self.subModel1.fit(X[splitIndex1], y[splitIndex1], thresholds=thresholds)
        self.subModel0 = DecisionTree(self.max_depth - 1, stump_class=self.stump_class)
        self.subModel0.fit(X[splitIndex0], y[splitIndex0], thresholds=thresholds)

    def predict(self, X):
        M, D = X.shape
        y = np.zeros(M)

        splitVariable = self.splitModel.splitVariable
        splitValue = self.splitModel.splitValue
        splitSat = self.splitModel.splitSat

        if splitVariable is None:
            # If no further splitting, return the majority label
            y = splitSat * np.ones(M)

        #the case with depth=1=a single stump
        elif self.subModel1 is None:
            return self.splitModel.predict(X)

        else:
            # Recurse sub-models
            j = splitVariable
            value = splitValue

            splitIndex1 = X[:, j] > value
            splitIndex0 = X[:, j] <= value

            y[splitIndex1] = self.subModel1.predict(X[splitIndex1])
            y[splitIndex0] = self.subModel0.predict(X[splitIndex0])

        return y


class RandomStumpGiniIndex(DecisionStumpGiniIndex):

        def fit(self, X, y, thresholds):
            # Randomly select k features.
            # Randomly permutes the feature indices, taking the first k
            D = X.shape[1]
            k = int(np.floor(np.sqrt(D)))

            chosen_features = np.random.choice(D, k, replace=False)

            DecisionStumpGiniIndex.fit(self, X, y, split_features=chosen_features, thresholds=thresholds)


"""**Random Tree**"""


class RandomTree(DecisionTree):

    def __init__(self, max_depth):
        DecisionTree.__init__(self, max_depth=max_depth, stump_class=RandomStumpGiniIndex)

    def fit(self, X, y, thresholds):
        N = X.shape[0]
        boostrap_inds = np.random.choice(N, N, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y, thresholds=thresholds)




"""**Random Forest**"""


class RandomForest:

    def __init__(self, num_trees, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.thresholds = None

    def fit(self, X, y):
        
        self.trees = []
        self.create_splits(X)
        for m in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
            tree.fit(X, y, thresholds=self.thresholds)
            self.trees.append(tree)

    def predict(self, X):
        t = X.shape[0]
        yhats = np.ones((t, self.num_trees), dtype=np.uint8)
        # Predict using each model
        for m in range(self.num_trees):
            yhats[:, m] = self.trees[m].predict(X)

        # Take the most common label
        return stats.mode(yhats, axis=1)[0].flatten()

    def create_splits(self, X):
        #k value obtained via elbow method
        N, D = X.shape
        splits = []
        for i in range(D):
            model = Kmeans(k=10)
            #all values in an example
            
            vec = X[:,i].reshape(N, 1)
            model.fit(vec)
            threshs = model.means
            splits.append(np.squeeze(threshs))
        
        self.thresholds = splits
        
   