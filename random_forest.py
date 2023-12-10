from decision_tree_regression import *
import numpy as np


class RandomForestRegressor:
    """ Random Forest Regressor """

    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=2):
        """
        Initializes the RandomForestRegressor with specified parameters.

        Parameters:
        - n_estimators (int): The number of trees in the forest.
        - min_samples_split (int): The minimum number of samples required to split an internal node.
        - max_depth (int): The maximum depth of the trees.

        Attributes:
        - trees (list): A list to store the individual decision trees.
        """
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        """
        Trains the Random Forest on the given dataset.

        This method creates individual decision trees, each trained on a bootstrap sample
        of the original dataset, and stores them in the 'trees' list.

        Parameters:
        - X: The features of the training dataset.
        - y: The target values of the training dataset.
        """
        for _ in range(self.n_estimators):
            # Initialize a new decision tree with the given parameters
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            # Generate a bootstrap sample of the dataset
            bootstrap_X, bootstrap_y = self._bootstrap_sample(X, y)
            # Fit the tree to the bootstrap sample
            tree.fit(bootstrap_X, bootstrap_y)
            # Add the fitted tree to the list of trees
            self.trees.append(tree)

    def predict(self, X):
        """
        Predicts target values for the given dataset using the trained Random Forest.

        The prediction for each instance is the average prediction of all the trees.

        Parameters:
        - X: The features of the dataset for which to make predictions.

        Returns:
        - The average predictions of all trees.
        """
        # Collect predictions from each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Return the average of these predictions
        return np.mean(tree_predictions, axis=0)

    @staticmethod
    def _bootstrap_sample(X, y):
        """
        Generates a bootstrap sample from the dataset.

        A bootstrap sample is a randomly selected subset of data with replacement.

        Parameters:
        - X: The features of the dataset.
        - y: The target values of the dataset.

        Returns:
        - A bootstrap sample of the dataset.
        """
        n_samples = X.shape[0]
        # Randomly select indices with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        # Return the corresponding sample
        return X[indices], y[indices]


def plot_combined_regression_with_random_forest(train_data, test_data, coefficients, m, q, tree_regressor, forest_regressor):
    """
    Plots the results of linear, polynomial, decision tree, and random forest regressions on the training and test data.
    """
    # Preparazione dei dati per il plotting
    X_train = train_data['Numerical_Index_scaled'].values
    X_test = test_data['Numerical_Index_scaled'].values

    # Calcoli per la regressione polinomiale
    y_pred_poly_train = sum(coeff * X_train ** i for i, coeff in enumerate(coefficients))
    y_pred_poly_test = sum(coeff * X_test ** i for i, coeff in enumerate(coefficients))

    # Calcoli per la regressione lineare
    y_pred_linear_train = m * X_train + q
    y_pred_linear_test = m * X_test + q

    print("1")
    # Predizioni con l'albero decisionale
    y_pred_tree_test = tree_regressor.predict(X_test.reshape(-1, 1))
    print("2")
    print(f"{forest_regressor}")
    print(f"{X_test.reshape(-1, 1)}")
    # Predizioni con la foresta casuale
    print("Dimensioni di X_test prima del reshape:", X_test.shape)
    X_test_reshaped = X_test.reshape(-1, 1)
    print("Dimensioni di X_test dopo il reshape:", X_test_reshaped.shape)
    print("Primi 5 elementi di X_test dopo il reshape:", X_test_reshaped[:5])
    for i, tree in enumerate(forest_regressor.trees):
        single_tree_pred = tree.predict(X_test_reshaped)
        print(f"Previsioni dell'albero {i}: {single_tree_pred[:5]}")

    y_pred_forest_test = forest_regressor.predict(X_test_reshaped)
    print("Previsioni della Foresta Casuale:", y_pred_forest_test[:5])

    y_pred_forest_test = forest_regressor.predict(X_test.reshape(-1, 1))
    print("3")
    # Ordinamento degli indici per il plotting
    sorted_indices = np.argsort(X_test)
    print("4")
    X_test_sorted = X_test[sorted_indices]
    print("5")
    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(train_data['Numerical_Index_scaled'], train_data['Close_scaled'], color="lightblue", label="Training Data", s=1)
    plt.scatter(test_data['Numerical_Index_scaled'], test_data['Close_scaled'], color="blue", label="Test Data", s=1)
    plt.plot(X_test_sorted, y_pred_poly_test[sorted_indices], color="red", label="Polynomial Regression")
    plt.plot(X_train, y_pred_linear_train, color="green", label="Linear Regression")
    plt.plot(X_test_sorted, y_pred_tree_test[sorted_indices], color='purple', label='Decision Tree Regression')
    plt.plot(X_test_sorted, y_pred_forest_test[sorted_indices], color='orange', label='Random Forest Regression')
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Combined Regression Models")
    plt.legend()
    plt.show()


def plot_all_regression_notWorking(train_data, test_data, coefficients, m, q, y_pred_tree, y_pred_forest):
    """
    Plots all regression results on a single graph.

    Parameters:
    - train_data: DataFrame containing training data.
    - test_data: DataFrame containing test data.
    - coefficients: Coefficients of the polynomial regression.
    - m, q: Parameters of the linear regression.
    - y_pred_tree: Predictions from the decision tree regressor.
    - y_pred_forest: Predictions from the random forest regressor.
    """
    # Preparazione dei dati per il plotting
    X_train = train_data['Numerical_Index_scaled'].values
    X_test = test_data['Numerical_Index_scaled'].values

    # Calcoli per la regressione polinomiale
    y_pred_poly_test = sum(coeff * X_test ** i for i, coeff in enumerate(coefficients))

    # Calcoli per la regressione lineare
    y_pred_linear_test = m * X_test + q

    # Ordinamento degli indici per il plotting
    sorted_indices = np.argsort(X_test)
    X_test_sorted = X_test[sorted_indices]

    # Assicurati che le dimensioni delle predizioni corrispondano
    if len(X_test) != len(y_pred_tree) or len(X_test) != len(y_pred_forest):
        print("Errore: le dimensioni delle predizioni non corrispondono con i dati di test.")
        return

    # Ordina le predizioni in base a X_test_sorted
    y_pred_poly_test_sorted = y_pred_poly_test[sorted_indices]
    y_pred_linear_test_sorted = y_pred_linear_test[sorted_indices]
    y_pred_tree_sorted = y_pred_tree[sorted_indices]
    y_pred_forest_sorted = y_pred_forest[sorted_indices]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(train_data['Numerical_Index_scaled'], train_data['Close_scaled'], color="lightblue", label="Training Data", s=1)
    plt.scatter(test_data['Numerical_Index_scaled'], test_data['Close_scaled'], color="blue", label="Test Data", s=1)
    plt.plot(X_test_sorted, y_pred_poly_test_sorted, color="red", label="Polynomial Regression")
    plt.plot(X_test_sorted, y_pred_linear_test_sorted, color="green", label="Linear Regression")
    plt.plot(X_test_sorted, y_pred_tree_sorted, color='purple', label='Decision Tree Regression')
    plt.plot(X_test_sorted, y_pred_forest_sorted, color='orange', label='Random Forest Regression')
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Combined Regression Models")
    plt.legend()
    plt.show()
