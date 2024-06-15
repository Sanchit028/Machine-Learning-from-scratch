import numpy as np

class LogisticRegression:
    ''' 
    Scratch implementation of Logistic Regression.
    '''
    def __init__(self, lr=0.001, n_iters=1000):
        '''
        Initialize the logistic regression model.

        Parameters:
        lr : float, optional (default=0.001)
            Learning rate for gradient descent.
        n_iters : int, optional (default=1000)
            Number of iterations for gradient descent.
        '''
        self.lr = lr
        self.n_iters = n_iters
        self.m=None
        self.b=0

    def mse(self, y, y_pred):
        ''' 
        Calculate the mean squared error between true values and predicted values.

        Parameters:
        y : numpy array, shape (n_samples,)
            True labels.
        y_pred : numpy array, shape (n_samples,)
            Predicted labels.

        Returns:
        float
            Mean squared error.
        '''
        return np.mean((y-y_pred)**2)

    def sigmoid(self, v):
        '''
        Sigmoid is the function that converts our linear regression code into logistic regression.
        It does so by setting a upper value to 1 and a lower to 0 and changing the values likewise.
        Parameters:
        v : numpy array
            Input to the sigmoid function.

        Returns:
        numpy array
            Output of the sigmoid function.
        '''
        return 1/(1+np.exp(-v))  #Formula of sigmoid

    def fit(self, X_train, y_train, print_cost=False):
        '''
        For the training of the model using gradient descent.

        Parameters:
        X_train : pandas DataFrame, shape (n_samples, n_features)
            Training data.
        y_train : pandas Series, shape (n_samples,)
            Target labels.
        print_cost : bool, optional (default=False)
            Whether to print the cost after each iteration.

        Returns:
        numpy array
            Coefficients (weights) of the model.
        float
            Intercept of the model.
        '''
        n=X_train.shape[0]
        self.m = np.zeros(X_train.shape[1])  # Initialize coefficients
        cost_prev=0
        X_train, y_train= X_train.values, y_train.values # Convert DataFrame to numpy array

        for _ in range(self.n_iters):

            y_Pred= self.sigmoid(np.dot(X_train, self.m) + self.b)
            cost=self.mse(y_train, y_Pred)  # Compute cost (mean squared error)

            if print_cost:
                #print the cost function
                print(f"Iteration: {_ + 1}\tCost: {cost:.10f}")
            if np.isclose(cost, cost_prev, rtol=1e-8):
                # Check convergence (stop if cost change is small)
                break

            # Partial Derivative updates for m and b
            dm=(1/n) * np.dot(X_train.T, y_Pred-y_train)
            db=(1/n) * np.sum(y_Pred-y_train)

            self.m=self.m - dm*self.lr # Update coefficients
            self.b=self.b - db*self.lr # Update intercept
            cost_prev=cost

        return self.m, self.b

    def predict(self, X):
        '''
        For the prediction of new data by using SIGMOID Function
        And the turning it into a categorical value.
        
        Parameters:
        X : pandas DataFrame, shape (n_samples, n_features)
            Input data for prediction.

        Returns:
        numpy array
            Predicted class labels (0 or 1).
        '''
        X=X.values # Convert DataFrame to numpy array
        prediction= self.sigmoid(np.dot(X, self.m) + self.b)
        categorical_prediction=[0 if i<=0.5 else 1 for i in prediction]
        return categorical_prediction
