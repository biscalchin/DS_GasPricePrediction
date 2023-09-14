import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl


def load_data():
    file_name = "1m_interval_NG_2023-09-13_7d_period.csv"
    folder_name = "Datas"
    path = folder_name + "/" + file_name
    data = pd.read_csv(path)
    return data


def linear_regression():
    data = load_data()
    print(data)
    plt.scatter(data.index, data.Close)
    plt.show()


if __name__ == '__main__':
    linear_regression()
