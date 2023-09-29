import numpy as np
from numba import cuda


@cuda.jit
def _gpu_fit(n_iters, lr, X, y, weights, bias):
    n_samples, n_features = X.shape
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdim = cuda.blockDim.x
    gdim = cuda.gridDim.x
    n_rep = n_iters // 20

    for i in range(n_iters):
        for j in range(bx, n_samples, bdim * gdim):
            y_pred = bias[0]
            for k in range(n_features):
                y_pred += X[j, k] * weights[k]

            db = y_pred - y[j]
            for k in range(n_features):
                dw = X[j, k] * db
                cuda.atomic.add(weights, k, -lr * dw)
            cuda.atomic.add(bias, 0, -lr * db)

        if i % n_rep == 0:
            if tx == 0 and bx == 0:
                print("Training Iteration ", i)  # Modified print statement


class LinearRegression:

    def __init__(self, lr=0.0000001, n_iters=10000):
        self.lr = lr  # Learning rate for gradient descent
        self.n_iters = n_iters  # Number of iterations for training
        self.weights = None  # Model weights (angular coefficients)
        self.bias = None  # Model bias (intercept)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.zeros(1)

        # Convert numpy arrays to numba device arrays
        X_device = cuda.to_device(X)
        y_device = cuda.to_device(y)
        weights_device = cuda.to_device(self.weights)
        bias_device = cuda.to_device(self.bias)

        # Define the number of threads in a block
        threads_per_block = 128
        # Calculate the number of blocks per grid
        blocks_per_grid = (n_samples + threads_per_block - 1) // threads_per_block

        # Launch the CUDA kernel
        _gpu_fit[blocks_per_grid, threads_per_block](self.n_iters, self.lr, X_device, y_device, weights_device,
                                                     bias_device)

        # Copy the trained weights and bias back to host
        self.weights = weights_device.copy_to_host()
        self.bias = bias_device.copy_to_host()

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
