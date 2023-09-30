from sklearn.model_selection import train_test_split
from data_scraper import *  # Import functions from data_scraper.py
from linear_Regression import *
from polinomial_Regression import *


def calculate_accuracy(mse):
    try:
        accuracy = (1 - mse) * 100  # as per your script
        return round(accuracy, 2)
    except Exception as e:
        print("Error in calculating accuracy: ", e)
        return None


def forecaster():
    try:

        # Load data using the load_data function
        print("Loading Data...")
        data = load_data()
        print(data)

        # Perform feature scaling on the data
        print("Performing Feature Scaling...")
        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        # Split the data into training and testing sets using train_test_split
        print("Splitting Data into Training and Testing Sets...")
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Define learning rate and number of iterations for gradient descent
        learning_rate = 0.001
        num_iterations = 1000

        # Perform linear regression using gradient descent and get the slope (m) and intercept (q)
        print("Performing Linear Regression using Gradient Descent...")
        try:
            print("Trying to compute using CUDA GPU technology...")
            m, q = gradient_descent_gpu(train_data, learning_rate, num_iterations)
        except Exception as e:
            print(e)
            print("Unable to compute using parallelization.")
            print("Using CPU computation: Warning! Process will take longer...")
            m, q = gradient_descent(train_data, learning_rate, num_iterations)

        # Calculate and print the Mean Squared Error on the test set for linear regression
        print("Calculating Mean Squared Error for Linear Regression on Test Set...")

        mse_linear = calculate_linear_mse(test_data, m, q)

        print(f"Mean Squared Error on Test Set for the linear regression is: {mse_linear}")

        # Perform polynomial regression with a degree of 16 and get the coefficients
        print("Performing Polynomial Regression...")

        degree = 16  # degree of the polynomial
        coefficients = polynomial_regression(train_data, degree)

        # Calculate and print the Mean Squared Error on the test set for polynomial regression
        print("Calculating Mean Squared Error for Polynomial Regression on Test Set...")

        mse_polynomial = calculate_polynomial_mse(test_data, coefficients)
        print(f"Mean Squared Error on Test Set for the polynomial regression is {mse_polynomial}")

        # For Linear Regression
        accuracy_linear = calculate_accuracy(mse_linear)
        print(f"Linear regression prediction accuracy: {accuracy_linear}%")

        # For Polynomial Regression
        accuracy_polynomial = calculate_accuracy(mse_polynomial)
        print(f"Polynomial regression prediction accuracy: {accuracy_polynomial}%")

        # Plot both linear and polynomial regression models along with the data
        print("Plotting Linear and Polynomial Regression Models along with the Data...")
        plot_combined_regression(train_data, test_data, coefficients, m, q)

    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print("Exception encountered:", e)


# Run the forecaster function if the script is executed as the main module
if __name__ == '__main__':
    forecaster()
