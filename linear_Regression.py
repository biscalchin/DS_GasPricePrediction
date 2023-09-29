import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from progress_bar import *

def feature_scaling(data):
    """
    This function takes a DataFrame 'data' and performs feature scaling on the 'Close' column and
    a newly created 'Numerical_Index' column. The scaled values are stored in new columns
    'Close_scaled' and 'Numerical_Index_scaled'.

    :param data: pandas DataFrame, expected to have a 'Close' column.
    :return: Modified pandas DataFrame with additional columns for scaled features.
    """

    # Create a new column 'Numerical_Index' in the DataFrame 'data'.
    # Assign a unique index (integer) to each row in the DataFrame.
    data['Numerical_Index'] = range(len(data))

    # Create a new column 'Numerical_Index_scaled' in the DataFrame 'data'.
    # Standardize the 'Numerical_Index' column by subtracting its mean and dividing by its standard deviation.
    data['Numerical_Index_scaled'] = (data['Numerical_Index'] - data['Numerical_Index'].mean()) / data[
        'Numerical_Index'].std()

    # Create a new column 'Close_scaled' in the DataFrame 'data'.
    # Standardize the 'Close' column by subtracting its mean and dividing by its standard deviation.
    data['Close_scaled'] = (data['Close'] - data['Close'].mean()) / data['Close'].std()

    # Return the modified DataFrame with the newly added and scaled columns.
    return data


def gradient_descent(data, learning_rate, num_iterations):
    m = 0
    q = 0
    n = len(data)

    for i in range(num_iterations):
        # Call progress_bar function here to show the progress
        progress_bar(i, num_iterations)

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

    # Print a new line after the progress bar is complete
    print()

    return m, q


def plot_data_linear_regression(data, m, q):
    plt.scatter(data['Numerical_Index_scaled'], data['Close_scaled'], color="blue")
    plt.plot(data['Numerical_Index_scaled'], m * data['Numerical_Index_scaled'] + q, color="red")
    plt.xlabel("Normalized Numerical Index")
    plt.ylabel("Normalized Close Price")
    plt.title("Linear Regression")
    plt.show()


def calculate_linear_mse(data, m, q):
    predictions = m * data['Numerical_Index_scaled'] + q
    mse = mean_squared_error(data['Close_scaled'], predictions)
    return mse

