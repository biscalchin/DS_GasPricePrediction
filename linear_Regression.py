import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from progress_bar import *
import cupy as cp


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
    # Initialize the parameters 'm' (slope) and 'q' (y-intercept) for linear regression
    m = 0
    q = 0
    # Determine the number of data points
    n = len(data)

    for i in range(num_iterations):
        # Update the progress bar for each iteration
        progress_bar(i+1, num_iterations)

        # Initialize gradients for 'm' and 'q' to zero for each iteration
        m_gradient = 0
        q_gradient = 0

        for j in range(n):
            # Extract the scaled numerical index and the scaled close value for each data point
            x = data['Numerical_Index_scaled'].iloc[j]
            y = data['Close_scaled'].iloc[j]
            # Calculate the current prediction based on the current values of 'm' and 'q'
            prediction = m * x + q

            # Compute the partial derivatives (gradients) of the cost function with respect to 'm' and 'q'
            m_gradient += -(2 / n) * x * (y - prediction)
            q_gradient += -(2 / n) * (y - prediction)

        # Update the values of 'm' and 'q' by moving against the gradient by a step size determined by the learning rate
        m -= learning_rate * m_gradient
        q -= learning_rate * q_gradient

    # Print a new line for readability after the progress bar is complete
    print()

    # Return the optimized values of 'm' and 'q' after completing all iterations
    return m, q



def gradient_descent_gpu(data, learning_rate, num_iterations):
    m = cp.zeros(1)
    q = cp.zeros(1)
    n = len(data)

    x_data = cp.array(data['Numerical_Index_scaled'].tolist())
    y_data = cp.array(data['Close_scaled'].tolist())

    for i in range(num_iterations):
        progress_bar(i + 1, num_iterations)

        prediction = m * x_data + q
        error = y_data - prediction

        m_gradient = -(2 / n) * cp.sum(x_data * error)
        q_gradient = -(2 / n) * cp.sum(error)

        m -= learning_rate * m_gradient
        q -= learning_rate * q_gradient

    print()
    return cp.asnumpy(m), cp.asnumpy(q)


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
