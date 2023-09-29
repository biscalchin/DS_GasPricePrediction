import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_data():
    file_name = "1m_interval_NG_2023-09-15_7d_period.csv"
    folder_name = "Datas"
    path = folder_name + "/" + file_name
    data = pd.read_csv(path)
    return data

def feature_scaling(data):
    y_mean = data['Close'].mean()
    y_std = data['Close'].std()
    data['Close_scaled'] = (data['Close'] - y_mean) / y_std
    return data

def linear_regression(data):
    X = data.index.values
    y = data['Close_scaled'].values

    n = len(X)

    # Calculate the coefficients (slope and intercept) for linear regression
    sum_X = np.sum(X)
    sum_y = np.sum(y)
    sum_Xy = np.sum(X * y)
    sum_X_squared = np.sum(X**2)

    slope = (n * sum_Xy - sum_X * sum_y) / (n * sum_X_squared - sum_X**2)
    intercept = (sum_y - slope * sum_X) / n

    return slope, intercept

def plot_data_and_regression(data, slope, intercept):
    X = data.index.values
    y_pred = slope * X + intercept

    plt.scatter(X, data['Close_scaled'], color="blue", label="Data")
    plt.plot(X, y_pred, color="red", label="Linear Regression")
    plt.xlabel("Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Linear Regression")
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

        slope, intercept = linear_regression(data)

        plot_data_and_regression(data, slope, intercept)

    except KeyboardInterrupt:
        print("Task interrupted successfully")
    except Exception as e:
        print("Exception encountered:", e)

if __name__ == '__main__':
    main()