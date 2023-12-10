import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from progress_bar import *
import seaborn as sns
import pandas as pd


def relu(x):
    """
    Implements the ReLU (Rectified Linear Unit) activation function.

    Parameters:
    - x: Input array or value.

    Returns:
    - An array where each element is the max of 0 and the element in x.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Computes the derivative of the ReLU function.

    Parameters:
    - x: Input array or value.

    Returns:
    - An array where each element is 1 if the corresponding element in x is greater than 0, otherwise 0.
    """
    return np.where(x > 0, 1, 0)


def leaky_relu(x, alpha=0.01):
    """
    Implements the Leaky ReLU activation function.

    Parameters:
    - x: Input array or value.
    - alpha: Slope coefficient for negative inputs.

    Returns:
    - An array where each element is alpha times the element if it's negative, and the element itself if it's positive.
    """
    return np.maximum(alpha * x, x)


def leaky_relu_derivative(x, alpha=0.01):
    """
    Computes the derivative of the Leaky ReLU function.

    Parameters:
    - x: Input array or value.
    - alpha: Slope coefficient for negative inputs.

    Returns:
    - An array where each element is alpha if the corresponding element in x is less than or equal to 0, otherwise 1.
    """
    return np.where(x > 0, 1, alpha)


class ImprovedNeuralNetwork:
    """
    A neural network implementation for regression tasks with Leaky ReLU activation and gradient descent optimization.

    Attributes:
    - layer_sizes (list): Sizes of each layer in the network.
    - learning_rate (float): Learning rate for optimization.
    - epochs (int): Number of epochs for training.
    - weights (list): The weights matrices of the network.
    - biases (list): The biases vectors of the network.
    - loss_history (list): The history of loss values during training.
    """

    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000):
        """
        Initializes the neural network.

        This constructor method sets up the initial parameters for the neural network,
        including the architecture (number and size of layers), learning rate, and number
        of epochs for training. It also initializes the weights and biases for each layer
        using a specific initialization strategy for better performance.

        Parameters:
        - layer_sizes (list): A list containing the sizes of each layer in the network.
          The length of this list determines the number of layers in the network, and
          each element specifies the number of neurons in that layer.
        - learning_rate (float): The learning rate used in the optimization algorithm.
          This value influences the step size in the gradient descent optimization.
        - epochs (int): The number of times the entire training dataset will be passed
          forward and backward through the neural network during training.

        Attributes:
        - weights (list): A list where each element is a weight matrix corresponding to
          connections between two layers. These matrices are initialized using a normal
          distribution scaled by the square root of the number of neurons in the
          preceding layer (He initialization).
        - biases (list): A list of bias vectors, one for each layer (excluding the input
          layer). These are initialized to zero.
        - loss_history (list): A list to record the loss value (such as mean squared error)
          after each epoch during training. This is useful for monitoring the training
          progress.
        """

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            # Initializing weights using He initialization for better convergence with
            # ReLU activations. This initialization helps in alleviating the vanishing
            # gradient problem in deep networks.
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])
            # Initializing biases to zero. In practice, initializing biases to zero is
            # usually fine regardless of the choice of activation function.
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.loss_history = []  # Initialize the list to store loss values per epoch

    def forward(self, X):
        """
        Performs forward propagation through the neural network.

        Parameters:
        - X: The input data as a numpy array. Each row represents a sample, and each column represents a feature.

        Returns:
        - zs: A list of the linear combinations (z values) for each layer.
        - activations: A list of the activations for each layer.

        The method calculates the linear combination (z) for each layer by multiplying the input/activation of the
        previous layer with the weights and adding the bias. It then applies the Leaky ReLU activation function to these
        linear combinations to get the activations of the current layer. These steps are repeated for each layer in the network.
        """

        # Initialize the activation for the input layer as the input data itself
        activation = X
        activations = [X]  # List to store all activations, layer by layer
        zs = []  # List to store all z vectors, layer by layer

        for w, b in zip(self.weights, self.biases):
            # Compute the linear combination of the current layer's weights, previous layer's activation, and bias
            z = np.dot(activation, w) + b
            # Store the computed linear combination
            zs.append(z)
            # Apply the Leaky ReLU activation function to the linear combination
            activation = leaky_relu(z)
            # Store the output activation of the current layer
            activations.append(activation)

        return zs, activations

    def backward(self, zs, activations, y):
        """
        Performs backward propagation through the neural network.

        Parameters:
        - zs: A list of the linear combinations (z values) for each layer as calculated in the forward pass.
        - activations: A list of the activations for each layer as calculated in the forward pass.
        - y: The actual target values.

        Returns:
        - nabla_w: Gradients (partial derivatives) of the cost function with respect to the weights.
        - nabla_b: Gradients (partial derivatives) of the cost function with respect to the biases.

        The method calculates the gradient of the cost function with respect to each parameter (weights and biases)
        in the network. This is done by applying the chain rule to propagate the error backward from the output layer
        to the input layer. The gradients are used to update the weights and biases in the training step.
        """

        # Calculate the initial delta (error) at the output layer
        delta = self.loss_derivative(activations[-1], y) * leaky_relu_derivative(zs[-1])
        # Calculate the gradient for biases at the last layer
        nabla_b = [np.sum(delta, axis=0, keepdims=True)]
        # Calculate the gradient for weights at the last layer
        nabla_w = [np.dot(activations[-2].T, delta)]

        # Iterate over the layers in reverse order starting from the second last layer
        for l in range(2, len(self.layer_sizes)):
            z = zs[-l]  # The linear combination at the current layer
            sp = leaky_relu_derivative(z)  # Derivative of activation function at the current layer
            delta = np.dot(delta, self.weights[-l + 1].T) * sp  # Update the error term (delta)
            nabla_b.insert(0, np.sum(delta, axis=0, keepdims=True))  # Calculate and store the gradient for biases
            nabla_w.insert(0, np.dot(activations[-l - 1].T, delta))  # Calculate and store the gradient for weights

        return nabla_w, nabla_b

    def update_params(self, nabla_w, nabla_b):
        """
        Updates the weights and biases of the network using gradient descent.

        Parameters:
        - nabla_w: Gradients of the cost function with respect to the weights.
        - nabla_b: Gradients of the cost function with respect to the biases.

        This method updates each weight and bias in the network by subtracting a portion of the gradient.
        The portion is determined by the learning rate and the scale of the gradient.
        """
        # Update weights with gradient descent
        self.weights = [w - (self.learning_rate / len(nabla_w)) * nw for w, nw in zip(self.weights, nabla_w)]
        # Update biases with gradient descent
        self.biases = [b - (self.learning_rate / len(nabla_b)) * nb for b, nb in zip(self.biases, nabla_b)]

    def train(self, X, y):
        """
        Trains the neural network on the provided dataset.

        Parameters:
        - X: Input features, a numpy array where each row is a sample and each column is a feature.
        - y: Target values, a numpy array corresponding to the input samples.

        The method iterates over the number of epochs, performing forward and backward propagation,
        and updating the network parameters in each iteration. It also records the training loss after each epoch.
        """
        for epoch in range(self.epochs):
            # Display the progress bar for training
            progress_bar(epoch + 1, self.epochs)
            # Forward propagation
            zs, activations = self.forward(X)
            # Calculate loss
            loss = mean_squared_error(y, activations[-1])
            # Record loss
            self.loss_history.append(loss)
            # Backward propagation to compute gradients
            nabla_w, nabla_b = self.backward(zs, activations, y)
            # Update network parameters
            self.update_params(nabla_w, nabla_b)

    def predict(self, X):
        """
        Makes predictions using the trained neural network.

        Parameters:
        - X: Input features, a numpy array where each row is a sample and each column is a feature.

        Returns:
        - The predictions of the neural network as a numpy array.

        This method uses forward propagation to compute the output of the network for the given input.
        """
        # Forward propagation to get the output activation
        _, activations = self.forward(X)
        return activations[-1]  # Return the final layer's activation as the prediction

    def loss_derivative(self, output_activations, y):
        """
        Computes the derivative of the loss function.

        Parameters:
        - output_activations: The activations (outputs) from the final layer of the network.
        - y: The actual target values.

        Returns:
        - The derivative of the loss function.

        This method calculates the gradient of the loss function with respect to the activations of the output layer.
        It is used during backpropagation to compute gradients for the output layer.
        """
        return output_activations - y  # Derivative of mean squared error loss

    def plot_loss(self):
        """
        Plots the training loss over each epoch.

        This method visualizes how the loss of the neural network decreases (ideally) over time during training.
        It is a useful tool for monitoring the training process and diagnosing issues with model learning.
        """
        plt.figure()
        plt.plot(self.loss_history)  # Plot the recorded loss over epochs
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()  # Display the plot

    def plot_predictions(self, X, y):
        """
        Plots the neural network's predictions against the actual data.

        Parameters:
        - X: Input features, used for making predictions.
        - y: Actual target values.

        This method provides a visual comparison between the predictions made by the neural network and the actual data.
        It is useful for assessing the model's performance visually.
        """
        y_pred = self.predict(X)  # Get predictions from the neural network
        plt.figure()
        plt.scatter(X[:, 0], y, label='True Data', color='blue', alpha=0.25)  # Plot actual data
        plt.scatter(X[:, 0], y_pred, label='Predictions', color='red', alpha=0.25)  # Plot predicted data
        plt.title('True Data vs Neural Network Predictions')
        plt.xlabel('Input Features')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.show()  # Display the plot
