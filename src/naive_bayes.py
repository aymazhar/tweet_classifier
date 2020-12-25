import numpy as np

class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape
        
        num_measures = 2
        num_classes = 2
        
        counts = np.bincount(y.astype(int))
        
        #initialize an array for the avg and var for each feature
        #p_0 represents class 0, p_1 represents class 1
        p_0av = np.ones((D, num_measures))
        p_1av = np.ones((D, num_measures))
        
        for i in range(D):
            for j in range(num_classes):
                
                avg = np.sum(X[y==j, i]) / counts[j]
                
                diff = np.array(X[y==j, i])
                sqd_diff = np.sum([(avg+(x*-1))**2 for x in diff]) / counts[j]
                var = np.sqrt(sqd_diff)
                if (j == 0):
                    p_0av[i][0] = avg
                    p_0av[i][1] = var
                else:
                    p_1av[i][0] = avg
                    p_1av[i][1] = var
                    
        self.p_0av = p_0av
        self.p_1av = p_1av
        
     
    
     
    def predict(self, X):
        #indices 0 = avg, 1 = var for p_0av & p_1av
        N, D = X.shape
        p_0av = self.p_0av
        p_1av = self.p_1av
        
        pi_term = np.sqrt(2*np.pi)
        
        y_pred = np.zeros(N)
        
        #for each feature in each example calc prob
        #sum up the total prob for that class, for that n
        for n in range(N):
            probs = np.zeros(2)
            
            for d in range(D):
                #class 0 calcs
                avg_0 = p_0av[d][0]
                var_0 = p_0av[d][1]
                
                term1 = 0.5*(((X[n,d] - avg_0)/var_0)**2)
                term2 = np.log(var_0 * pi_term)
                pr0 = term1 + term2
                probs[0] -= pr0
           
                #class 1 calcs
                avg_1 = p_1av[d][0]
                var_1 = p_1av[d][1]
                
                term1 = 0.5*(((X[n,d] - avg_1)/var_1)**2)
                term2 = np.log(var_1 * pi_term)
                pr1 = term1 + term2
                probs[1] -= pr1
            
            #set y_pred for each n to class with higher prob 
            y_pred[n] = np.argmax(probs)

        return y_pred
