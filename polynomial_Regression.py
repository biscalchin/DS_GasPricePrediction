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
    # Extract the scaled numerical index and the scaled closing values from the data
    X = data['Numerical_Index_scaled'].values
    y = data['Close_scaled'].values

    # Initialize matrix A and vector B for the linear system of equations
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)

    for i in range(degree + 1):
        # Update the progress bar for each step in the outer loop
        progress_bar(i, degree)
        for j in range(degree + 1):
            # Compute each element of matrix A as the sum of X raised to the power (i+j)
            A[i][j] = np.sum(X ** (i + j))
        # Compute each element of vector B as the sum of X raised to the power i times y
        B[i] = np.sum(X ** i * y)

    # Solve the linear system of equations A * coefficients = B to find the polynomial coefficients
    coefficients = np.linalg.solve(A, B)

    # Print a new line for better readability after the progress bar is complete
    print()

    # Return the calculated polynomial coefficients
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
