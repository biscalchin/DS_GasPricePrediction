import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    file_name = "1m_interval_NG_2023-09-15_7d_period.csv"
    folder_name = "Datas"
    path = folder_name + "/" + file_name
    data = pd.read_csv(path)
    return data


def feature_scaling(data):
    # Create a numerical index column for scaling
    data['Numerical_Index'] = range(len(data))

    # Scale the 'Numerical_Index' and 'Close' columns
    data['Numerical_Index_scaled'] = (data['Numerical_Index'] - data['Numerical_Index'].mean()) / data[
        'Numerical_Index'].std()
    data['Close_scaled'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

    return data


def gradient_descent(data, learning_rate, num_iterations):
    m = 0
    q = 0
    n = len(data)

    for i in range(num_iterations):
        m_gradient = 0
        q_gradient = 0

        for j in range(n):
            x = data['Numerical_Index_scaled'].iloc[j]
            y = data['Close_scaled'].iloc[j]
            prediction = m * x + q

            # Compute the gradients of the cost function with respect to m and q
            m_gradient += -(2 / n) * x * (y - prediction)
            q_gradient += -(2 / n) * (y - prediction)

        # Update m and q using the gradients and the learning rate
        m -= learning_rate * m_gradient
        q -= learning_rate * q_gradient

    return m, q


def plot_data_and_regression(data, m, q):
    plt.scatter(data['Numerical_Index_scaled'], data['Close_scaled'], color="blue")
    plt.plot(data['Numerical_Index_scaled'], m * data['Numerical_Index_scaled'] + q, color="red")
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Linear Regression")
    plt.show()


def main():
    try:
        data = load_data()
        print("Original Data:")
        print(data)

        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        learning_rate = 0.001
        num_iterations = 1000
        m, q = gradient_descent(data, learning_rate, num_iterations)

        print(f"Final slope (m): {m}")
        print(f"Final intercept (q): {q}")

        plot_data_and_regression(data, m, q)

    except KeyboardInterrupt:
        print("Task interrupted successfully")
    except Exception as e:
        print("Exception encountered:", e)


if __name__ == '__main__':
    main()