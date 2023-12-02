import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from progress_bar import *

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


class ImprovedNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.loss_history = []

    def forward(self, X):
        activation = X
        activations = [X]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = leaky_relu(z)
            activations.append(activation)

        return zs, activations

    def backward(self, zs, activations, y):
        delta = self.loss_derivative(activations[-1], y) * leaky_relu_derivative(zs[-1])
        nabla_b = [np.sum(delta, axis=0, keepdims=True)]
        nabla_w = [np.dot(activations[-2].T, delta)]

        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]
            sp = leaky_relu_derivative(z)
            delta = np.dot(delta, self.weights[-l + 1].T) * sp
            nabla_b.insert(0, np.sum(delta, axis=0, keepdims=True))
            nabla_w.insert(0, np.dot(activations[-l - 1].T, delta))

        return nabla_w, nabla_b

    def update_params(self, nabla_w, nabla_b):
        self.weights = [w - (self.learning_rate / len(nabla_w)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / len(nabla_b)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def train(self, X, y):
        for epoch in range(self.epochs):
            progress_bar(epoch + 1, self.epochs)
            zs, activations = self.forward(X)
            loss = mean_squared_error(y, activations[-1])
            self.loss_history.append(loss)
            nabla_w, nabla_b = self.backward(zs, activations, y)
            self.update_params(nabla_w, nabla_b)

            """if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
"""
    def predict(self, X):
        _, activations = self.forward(X)
        return activations[-1]

    def loss_derivative(self, output_activations, y):
        return output_activations - y

    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_history)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    def plot_predictions(self, X, y):
        y_pred = self.predict(X)
        plt.figure()
        plt.scatter(X[:, 0], y, label='True Data', color='blue', alpha=0.25)
        plt.scatter(X[:, 0], y_pred, label='Predictions', color='r', alpha=0.25)
        plt.title('True Data vs Neural Network Predictions')
        plt.xlabel('Input Features')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.show()

