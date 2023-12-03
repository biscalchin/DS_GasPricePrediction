import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from progress_bar import *


def calculate_polynomial_mse(data, coefficients):
    X = data['Numerical_Index_scaled'].values
    y_true = data['Close_scaled'].values
    y_pred = np.zeros_like(X)

    for i, coeff in enumerate(coefficients):
        y_pred += coeff * X ** i

    mse = mean_squared_error(y_true, y_pred)
    return mse


def polynomial_regression(data, degree):
    X = data['Numerical_Index_scaled'].values
    y = data['Close_scaled'].values
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)

    for i in range(degree + 1):
        # Call progress_bar function here to show the progress
        progress_bar(i, degree)
        for j in range(degree + 1):
            A[i][j] = np.sum(X ** (i + j))
        B[i] = np.sum(X ** i * y)

    coefficients = np.linalg.solve(A, B)
    # Print a new line after the progress bar is complete
    print()
    return coefficients


def plot_data_and_regression(data, coefficients):
    X = data['Numerical_Index_scaled'].values
    y_pred = np.zeros_like(X)

    for i, coeff in enumerate(coefficients):
        y_pred += coeff * X ** i

    # Sort X and y_pred for plotting
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    y_pred_sorted = y_pred[sorted_indices]

    plt.scatter(data['Numerical_Index_scaled'], data['Close_scaled'], color="blue", label="Data")
    plt.plot(X_sorted, y_pred_sorted, color="red", label="Polynomial Regression")
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Polynomial Regression")
    plt.legend()
    plt.show()


def plot_combined_regression(train_data, test_data, coefficients, m, q):
    X_train = train_data['Numerical_Index_scaled'].values
    y_pred_poly_train = np.zeros_like(X_train)

    X_test = test_data['Numerical_Index_scaled'].values
    y_pred_poly_test = np.zeros_like(X_test)

    for i, coeff in enumerate(coefficients):
        y_pred_poly_train += coeff * X_train ** i
        y_pred_poly_test += coeff * X_test ** i

    # Sort X and y_pred for plotting
    sorted_indices_train = np.argsort(X_train)
    X_train_sorted = X_train[sorted_indices_train]
    y_pred_poly_train_sorted = y_pred_poly_train[sorted_indices_train]

    sorted_indices_test = np.argsort(X_test)
    X_test_sorted = X_test[sorted_indices_test]
    y_pred_poly_test_sorted = y_pred_poly_test[sorted_indices_test]

    # Plotting
    plt.scatter(train_data['Numerical_Index_scaled'], train_data['Close_scaled'], color="lightblue", label="Training Data", s=1)
    plt.scatter(test_data['Numerical_Index_scaled'], test_data['Close_scaled'], color="blue", label="Test Data", s=1)
    plt.plot(X_train_sorted, y_pred_poly_train_sorted, color="red", label="Polynomial Regression")
    plt.plot(train_data['Numerical_Index_scaled'], m * train_data['Numerical_Index_scaled'] + q, color="green", label="Linear Regression")
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Combined Regression")
    plt.legend()
    plt.show()


def plot_combined_regression_v2(train_data, test_data, coefficients, m, q, y_pred_tree):
    X_train = train_data['Numerical_Index_scaled'].values
    y_pred_poly_train = np.zeros_like(X_train)

    X_test = test_data['Numerical_Index_scaled'].values
    y_pred_poly_test = np.zeros_like(X_test)

    # Calcolo delle predizioni per la regressione polinomiale
    for i, coeff in enumerate(coefficients):
        y_pred_poly_train += coeff * X_train ** i
        y_pred_poly_test += coeff * X_test ** i

    # Ordinamento degli array per il plotting
    sorted_indices_train = np.argsort(X_train)
    X_train_sorted = X_train[sorted_indices_train]
    y_pred_poly_train_sorted = y_pred_poly_train[sorted_indices_train]
    sorted_indices_test = np.argsort(X_test)
    X_test_sorted = X_test[sorted_indices_test]
    y_pred_poly_test_sorted = y_pred_poly_test[sorted_indices_test]

    # Ordinamento dei risultati dell'albero decisionale
    sorted_indices_tree = np.argsort(X_test)
    X_test_tree_sorted = X_test[sorted_indices_tree]
    y_pred_tree_sorted = y_pred_tree[sorted_indices_tree]

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(train_data['Numerical_Index_scaled'], train_data['Close_scaled'], color="lightblue", label="Training Data", s=1)
    plt.scatter(test_data['Numerical_Index_scaled'], test_data['Close_scaled'], color="blue", label="Test Data", s=1)
    plt.plot(X_train_sorted, y_pred_poly_train_sorted, color="red", label="Polynomial Regression")
    plt.plot(train_data['Numerical_Index_scaled'], m * train_data['Numerical_Index_scaled'] + q, color="green", label="Linear Regression")
    plt.plot(X_test_tree_sorted, y_pred_tree_sorted, color='purple', label='Decision Tree Regression')

    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Combined Regression Models")
    plt.legend()
    plt.show()
