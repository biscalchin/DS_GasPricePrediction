import numpy as np
import matplotlib.pyplot as plt
from progress_bar import *


class DecisionTreeNode:
    """ Decision Tree Node """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        """
        Initializes a node in the decision tree.

        Parameters:
        - feature_index (int): Index of the feature used for splitting the data.
        - threshold (float): The threshold value for splitting the data at the feature.
        - left (DecisionTreeNode): The left child node (for values less than or equal to the threshold).
        - right (DecisionTreeNode): The right child node (for values greater than the threshold).
        - value (float): The predicted value if this is a leaf node.
        """
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child
        self.right = right  # Right child
        self.value = value  # Predicted value if it's a leaf node

class DecisionTreeRegressor:
    """ Base implementation of the regression using Decision Tree """

    def __init__(self, min_samples_split=2, max_depth=2):
        """
        Initializes the DecisionTreeRegressor.

        Parameters:
        - min_samples_split (int): The minimum number of samples required to consider a split.
        - max_depth (int): The maximum depth of the tree.
        """
        self.root = None  # Root node of the decision tree
        self.min_samples_split = min_samples_split  # Minimum number of samples to split
        self.max_depth = max_depth  # Maximum depth of the tree

    def fit(self, X, y):
        """
        Trains the Decision Tree on the provided dataset.

        Parameters:
        - X: Feature data for training.
        - y: Target values.
        """
        self.root = self._build_tree(X, y)  # Start building the tree from the root

    def _build_tree(self, X, y, current_depth=0):
        """
        Recursively builds the decision tree.

        Parameters:
        - X: The feature data for the current node.
        - y: The target values for the current node.
        - current_depth (int): The current depth in the tree.

        Returns:
        - A DecisionTreeNode representing either an internal node or a leaf.
        """
        num_samples, num_features = X.shape

        # Check stopping criteria
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Find the best split
            best_split = self._get_best_split(X, y, num_samples, num_features)
            if best_split["value"] is not None:
                # Build the left and right subtrees recursively
                left_subtree = self._build_tree(best_split["dataset_left"][:, :-1], best_split["dataset_left"][:, -1], current_depth + 1)
                right_subtree = self._build_tree(best_split["dataset_right"][:, :-1], best_split["dataset_right"][:, -1], current_depth + 1)
                return DecisionTreeNode(feature_index=best_split["feature_index"], threshold=best_split["threshold"], left=left_subtree, right=right_subtree)

        # Create a leaf node
        return DecisionTreeNode(value=np.mean(y))

    def _get_best_split(self, X, y, num_samples, num_features):
        """
        Finds the best feature and threshold to split the data.

        Parameters:
        - X: Feature data.
        - y: Target values.
        - num_samples (int): The number of samples in the data.
        - num_features (int): The number of features in the data.

        Returns:
        - A dictionary containing information about the best split.
        """
        best_split = {}
        min_error = float("inf")

        for feature_index in range(num_features):
            for threshold in np.unique(X[:, feature_index]):
                dataset_left, dataset_right = self._split_dataset(X, y, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y_left, y_right = dataset_left[:, -1], dataset_right[:, -1]
                    current_error = self._calculate_mse(y_left, y_right)
                    if current_error < min_error:
                        min_error = current_error
                        best_split = {"feature_index": feature_index, "threshold": threshold, "dataset_left": dataset_left, "dataset_right": dataset_right, "value": min_error}

        return best_split

    @staticmethod
    def _split_dataset(X, y, feature_index, threshold):
        """
        Splits the dataset based on the given feature and threshold.

        Parameters:
        - X: Feature data.
        - y: Target values.
        - feature_index (int): The index of the feature to split on.
        - threshold (float): The value of the threshold for splitting.

        Returns:
        - Two datasets corresponding to the left and right splits.
        """
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        dataset_left = X[left_indices]
        dataset_right = X[right_indices]

        y_left = y[left_indices]
        y_right = y[right_indices]

        return np.column_stack((dataset_left, y_left)), np.column_stack((dataset_right, y_right))

    @staticmethod
    def _calculate_mse(y_left, y_right):
        """
        Calculates the mean squared error for a given split.

        Parameters:
        - y_left: Target values in the left split.
        - y_right: Target values in the right split.

        Returns:
        - The mean squared error of the split.
        """
        n_left, n_right = len(y_left), len(y_right)
        mse_left, mse_right = np.mean((y_left - np.mean(y_left)) ** 2), np.mean((y_right - np.mean(y_right)) ** 2)
        return (n_left * mse_left + n_right * mse_right) / (n_left + n_right)

    def predict(self, X):
        """
        Predicts values for the given data using the trained Decision Tree.

        Parameters:
        - X: The feature data for prediction.

        Returns:
        - An array of predicted values.
        """
        return np.array([self._predict_value(x, self.root) for x in X])

    def _predict_value(self, x, tree):
        """
        Recursively predicts a value by traversing the tree.

        Parameters:
        - x: A single instance of feature data.
        - tree: The current node in the decision tree.

        Returns:
        - The predicted value.
        """
        if tree.value is not None:
            return tree.value  # Return value if it's a leaf node
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict_value(x, tree.left)  # Go to left subtree
        else:
            return self._predict_value(x, tree.right)  # Go to right subtree


def plot_combined_regression_with_decision_tree(train_data, test_data, coefficients, m, q, tree_regressor):
    """
    Plots the results of linear, polynomial and decision tree regressions on the training and test data.
    """
    # Calculations for polynomial regression
    X_train = train_data['Numerical_Index_scaled'].values
    y_pred_poly_train = np.zeros_like(X_train)
    X_test = test_data['Numerical_Index_scaled'].values
    y_pred_poly_test = np.zeros_like(X_test)

    for i, coeff in enumerate(coefficients):
        y_pred_poly_train += coeff * X_train ** i
        y_pred_poly_test += coeff * X_test ** i

    # Index sorting for plotting polynomial regression
    sorted_indices_train = np.argsort(X_train)
    X_train_sorted = X_train[sorted_indices_train]
    y_pred_poly_train_sorted = y_pred_poly_train[sorted_indices_train]

    sorted_indices_test = np.argsort(X_test)
    X_test_sorted = X_test[sorted_indices_test]
    y_pred_poly_test_sorted = y_pred_poly_test[sorted_indices_test]

    # Calcoli per la regressione lineare
    y_pred_linear_train = m * train_data['Numerical_Index_scaled'] + q
    y_pred_linear_test = m * test_data['Numerical_Index_scaled'] + q

    # Addestramento e predizione con l'albero decisionale
    tree_regressor.fit(X_train.reshape(-1, 1), train_data['Close_scaled'])
    y_pred_tree_test = tree_regressor.predict(X_test.reshape(-1, 1))

    # Ordinamento degli indici per il plotting della regressione ad albero
    sorted_indices_tree = np.argsort(X_test)
    X_test_tree_sorted = X_test[sorted_indices_tree]
    y_pred_tree_test_sorted = y_pred_tree_test[sorted_indices_tree]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(train_data['Numerical_Index_scaled'], train_data['Close_scaled'], color="lightblue", label="Training Data", s=1)
    plt.scatter(test_data['Numerical_Index_scaled'], test_data['Close_scaled'], color="blue", label="Test Data", s=1)
    plt.plot(X_train_sorted, y_pred_poly_train_sorted, color="red", label="Polynomial Regression")
    plt.plot(X_train, y_pred_linear_train, color="green", label="Linear Regression")
    plt.plot(X_test_tree_sorted, y_pred_tree_test_sorted, color='purple', label='Decision Tree Regression')
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Combined Regression Models")
    plt.legend()
    plt.show()

