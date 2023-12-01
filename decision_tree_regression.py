import numpy as np
import matplotlib.pyplot as plt
from progress_bar import *


class DecisionTreeNode:
    """ Decision Tree Node """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # per decisione
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

        # per foglia
        self.value = value


class DecisionTreeRegressor:
    """ Base implementation of the regression using Decision Tree """

    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def fit(self, X, y):
        """ Train Decision Tree """
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive Decision Tree Builder """
        num_samples, num_features = X.shape
        progress_bar(current_depth, self.max_depth)

        # Stopping Criteria
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Find best split
            best_split = self._get_best_split(X, y, num_samples, num_features)
            if best_split["value"] is not None:
                # build subtree
                left_subtree = self._build_tree(best_split["dataset_left"][:, :-1], best_split["dataset_left"][:, -1], current_depth + 1)
                right_subtree = self._build_tree(best_split["dataset_right"][:, :-1], best_split["dataset_right"][:, -1], current_depth + 1)
                return DecisionTreeNode(feature_index=best_split["feature_index"], threshold=best_split["threshold"], left=left_subtree, right=right_subtree)

        # Leaf Node
        return DecisionTreeNode(value=np.mean(y))

    def _get_best_split(self, X, y, num_samples, num_features):
        """ Find the best division point for a node """
        best_split = {}
        min_error = float("inf")
        total_splits = num_features * len(np.unique(X))
        current_split = 0

        # checks every feature and every possible threshold value
        for feature_index in range(num_features):
            for threshold in np.unique(X[:, feature_index]):
                # Progress bar
                current_split += 1
                progress_bar(current_split, total_splits)

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
        """ Divides the dataset according to the threshold on a feature """
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold

        dataset_left = X[left_indices]
        dataset_right = X[right_indices]

        y_left = y[left_indices]
        y_right = y[right_indices]

        return np.column_stack((dataset_left, y_left)), np.column_stack((dataset_right, y_right))

    @staticmethod
    def _calculate_mse(y_left, y_right):
        """ Calculate the mean square error for the split """
        n_left, n_right = len(y_left), len(y_right)
        mse_left, mse_right = np.mean((y_left - np.mean(y_left)) ** 2), np.mean((y_right - np.mean(y_right)) ** 2)
        return (n_left * mse_left + n_right * mse_right) / (n_left + n_right)

    def predict(self, X):
        """ Predicts values for the data provided """
        return np.array([self._predict_value(x, self.root) for x in X])

    def _predict_value(self, x, tree):
        """ Performs recursive prediction by traversing the tree """
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._predict_value(x, tree.left)
        else:
            return self._predict_value(x, tree.right)


def plot_combined_regression_with_decision_tree(train_data, test_data, coefficients, m, q, regressor):
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
    regressor.fit(X_train.reshape(-1, 1), train_data['Close_scaled'])
    y_pred_tree_test = regressor.predict(X_test.reshape(-1, 1))

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
