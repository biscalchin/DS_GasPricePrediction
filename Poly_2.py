import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():
    file_name = "1m_interval_NG_2023-09-15_7d_period.csv"
    folder_name = "Old_Datas"
    path = folder_name + "/" + file_name
    data = pd.read_csv(path)
    return data


def calculate_mse(data, coefficients):
    X = data['Numerical_Index_scaled'].values
    y_true = data['Close_scaled'].values
    y_pred = np.zeros_like(X)

    for i, coeff in enumerate(coefficients):
        y_pred += coeff * X ** i

    mse = mean_squared_error(y_true, y_pred)
    return mse


def feature_scaling(data):
    # Create a numerical index column for scaling
    data['Numerical_Index'] = range(len(data))

    # Scale the 'Numerical_Index' and 'Close' columns
    data['Numerical_Index_scaled'] = (data['Numerical_Index'] - data['Numerical_Index'].mean()) / data[
        'Numerical_Index'].std()
    data['Close_scaled'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

    return data


def polynomial_regression(data, degree):
    X = data['Numerical_Index_scaled'].values
    y = data['Close_scaled'].values

    n = len(X)
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i][j] = np.sum(X ** (i + j))

        B[i] = np.sum(X ** i * y)

    coefficients = np.linalg.solve(A, B)

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



def main():
    try:
        data = load_data()
        print("Original Data:")
        print(data)

        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        degree = 16  # degree of the polynomial
        coefficients = polynomial_regression(train_data, degree)

        mse = calculate_mse(test_data, coefficients)
        print("Mean Squared Error on Test Set:", mse)

        print("Training Data and Regression:")
        plot_data_and_regression(train_data, coefficients)



    except KeyboardInterrupt:
        print("Task interrupted successfully")
    except Exception as e:
        print("Exception encountered:", e)


if __name__ == '__main__':
    main()
