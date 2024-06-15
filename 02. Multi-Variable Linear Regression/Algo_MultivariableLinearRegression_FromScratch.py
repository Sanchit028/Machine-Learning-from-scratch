import numpy as np

class MultiLR():
    # Initialization function
    def  __init__(self, lr = 0.001, n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.coefs=[]
        self.costs = []

    def predict(self, X):
        return X @ self.coefs  #Matrix Multiplication of Coefficients and the test dataset

    def cost(self, X, y):
        return ((y - (X @ self.coefs))**2).mean()  #Mean Square Error
        # MSE Formula = Mean(y - (X * Co-efficients)^2)

    def fit(self, X, y, print_cost = False):
        M, N = X.shape
        
        X = np.insert(X, 0, 1, axis = 1) # Concatenating the allOnes column to X_train(for the intercept value).

        np.random.seed(123)
        self.coefs = np.random.uniform(-10.0, 10.0, N + 1) #Generating a random intercept and coefficient value

        for x in range(self.n_iters):
            cost_ = self.cost(X, y)
            self.costs.append(cost_)
            if print_cost:
                print("Iteration :", x + 1, '\t', "Cost : " + '%.4f'%cost_)
            slope_array = np.zeros(N + 1)
            for i in range(M):
                f_xi = (self.coefs * X[i]).sum()
                y_i = y[i]
                for j in range(N + 1):
                    slope_array[j] += (-2/M) * (y_i - f_xi) * X[i,j]

            self.coefs -= (self.lr * slope_array)
        return self.coefs, self.costs 