import numpy as np

class LinearRegression:
    
    def __init__(self, lr=0.001, n_iters=1000):
        #Initializing the class
        self.lr = lr
        self.n_iters = n_iters
        self.m=0
        self.b=0
    
    @staticmethod    
    def mse(y, y_pred):
        # Calculates the mean square error between y and y predicted
        return np.mean((y-y_pred)**2)
    
    @staticmethod
    def train_test_split(X, y, test_size=0.2, random_state=None):
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        # Shuffle the indices
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        # Calculate the number of samples for the test set
        test_samples = int(test_size * len(X))

        # Split the indices into train and test sets
        test_indices = indices[:test_samples]
        train_indices = indices[test_samples:]

        # Split the data into train and test sets
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def fit(self, X_train, y_train):
        # Converting the pandas data frame
        self.m=0
        self.b=0
        N=X_train.shape[0]
        cost_prev=0
        
        for _ in range(self.n_iters):
            Y_Pred= (X_train * self.m) + self.b
            
            cost=self.mse(y_train, Y_Pred)
            if np.isclose(cost, cost_prev, rtol=1e-8): # Checks if the difference betweeen last error and this times error is less than 10^-8
                break
            
            # Partial Derivatives for m and b
            dm=-(2/N) * np.dot(X_train.T, y_train-Y_Pred)
            db=-(2/N) * np.sum(y_train-Y_Pred)
            
            #updating m and b
            self.m=self.m - dm*self.lr
            self.b=self.b - db*self.lr
            cost_prev=cost
        
        return self.m, self.b
            
    def predict(self, X):
        return ((X.dot(self.m)) + self.b)
