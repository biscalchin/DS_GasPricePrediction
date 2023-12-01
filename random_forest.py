from decision_tree_regression import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RandomForestRegressor:
    """ Random Forest Regressor """

    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """ Train Random Forest """
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            bootstrap_X, bootstrap_y = self._bootstrap_sample(X, y)
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        """ Predicts values for the data provided using the average of all trees """
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_predictions, axis=0)

    @staticmethod
    def _bootstrap_sample(X, y):
        """ Create a bootstrap sample of the dataset """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]


