import csv
import matplotlib.pyplot as plt
import numpy as np

# Load data from a CSV file
def load_data():
    file_name = "1m_interval_NG_2023-09-15_7d_period.csv"
    folder_name = "Datas"
    path = folder_name + "/" + file_name
    X, y = [], []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            X.append(float(row[3]))  # Assuming 'Close' is the column of interest
    return X

# Perform polynomial regression
def polynomial_regression(X, y, degree):
    n = len(X)
    A = np.zeros((degree + 1, degree + 1))
    B = np.zeros(degree + 1)

    for i in range(degree + 1):
        for j in range(degree + 1):
            A[i][j] = np.sum(X ** (i + j))
        B[i] = np.sum(X ** i * y)

    coefficients = np.linalg.solve(A, B)
    return coefficients

# Calculate polynomial values
def calculate_polynomial(coefficients, X):
    degree = len(coefficients) - 1
    y_pred = np.zeros(len(X))
    for i in range(degree + 1):
        y_pred += coefficients[i] * X ** (degree - i)
    return y_pred

# Plot data and regression line
def plot_data_and_regression(X, y, y_pred):
    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label="Polynomial Regression")
    plt.xlabel("Index")
    plt.ylabel("Close Price")
    plt.title("Polynomial Regression")
    plt.legend()
    plt.show()

def main():
    try:
        X = np.array(load_data())
        y = np.array(X)  # You can replace this with the actual target variable
        degree = 3  # Adjust the degree of the polynomial as needed

        coefficients = polynomial_regression(X, y, degree)
        y_pred = calculate_polynomial(coefficients, X)

        plot_data_and_regression(X, y, y_pred)

    except KeyboardInterrupt:
        print("Task interrupted successfully")
    except Exception as e:
        print("Exception encountered:", e)

if __name__ == '__main__':
    main()
