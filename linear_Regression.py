import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from class_l_reg import LinearRegression
from sklearn.model_selection import train_test_split


def load_data():
    file_name = "dataset.csv"
    data = pd.read_csv(file_name)
    return data


def mean_sq_err(y_test, prediction):
    return np.mean((y_test - prediction)**2)


def app():
    try:
        dataset = load_data()
        print(dataset)
        X = dataset.index.values.reshape(-1, 1)  # Reshape to make it a 2D array
        y = dataset["Close"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


        reg = LinearRegression(lr=0.0000001, n_iters=1000)
        reg.fit(X, y)
        prediction = reg.predict(X_test)
        mse = mean_sq_err(y_test, prediction)
        print(mse)
        y_pred_line = reg.predict(X)
        cmap = plt.get_cmap('viridis')
        fig = plt.figure(figsize=(8,6))
        m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=1)
        m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=1)
        plt.plot(X, y_pred_line, color="red", linewidth=1, label='Prediction')
        plt.show()


    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")
    except Exception as e:
        print("Exception encountered:", e)


if __name__ == '__main__':
    app()
