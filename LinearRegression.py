import numpy as np


class LinearRegression:

    def __init__(self, lr=0.0000001, n_iters=10000):
        # Initialize the Linear Regression model with hyperparameters
        self.lr = lr  # Learning rate for gradient descent
        self.n_iters = n_iters  # Number of iterations for training
        self.weights = None  # Model weights (angular coefficients)
        self.bias = None  # Model bias (intercept)

    def fit(self, X, y):
        # Fit the linear regression model to the given data

        # Get the number of samples and features in the input data
        n_samples, n_features = X.shape

        # Initialize the weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Define a variable to control the printing of training progress
        n_rep = self.n_iters / 20

        # Iterate through the training process
        for i in range(self.n_iters):
            # Make predictions using the current weights and bias
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate the gradient of the loss function with respect to weights and bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # Update the weights and bias using gradient descent
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

            # Print training progress every 'n_rep' iterations
            if i % n_rep == 0:
                print(f"Training Iteration {i}")
                print(f"dw = {dw}\ndb = {db}")

    def predict(self, X):
        # Make predictions using the trained linear regression model
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
