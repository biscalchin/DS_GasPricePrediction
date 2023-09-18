import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    file_name = "1m_interval_NG_2023-09-15_7d_period.csv"
    folder_name = "Datas"
    path = folder_name + "/" + file_name
    data = pd.read_csv(path)
    return data


def gradient_incline(m_now, q_now, datas, learning_rate):
    m_grad = 0
    q_grad = 0
    n = len(datas)
    for i in range(n):
        x = float(datas.index[i])
        y = datas.iloc[i].Close
        m_grad += -(2/n) * x * (y - (m_now * x + q_now))
        q_grad += -(2 / n) * (y - (m_now * x + q_now))
    m = m_now - m_grad * learning_rate
    q = q_now - q_grad * learning_rate
    return m, q


def linear_regression():
    try:
        data = load_data()
        print(data)
        m = 1
        q = 0
        lr = 0.0001
        iteration = 80
        for i in range(iteration):
            if i % 10 == 0:
                print(f"iteration: {i}")
            m, q = gradient_incline(m, q, data, lr)

        plt.scatter(data.index, data.Close, color="blue")
        plt.plot(list(range(0, 7000)), [m * x + q for x in range(0, 7000)], color="red")
        plt.show()

    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task interrupted successfully")
    # Handle any other exception and print its message
    except Exception as e:
        print("Exception encountered:", e)


if __name__ == '__main__':
    linear_regression()
