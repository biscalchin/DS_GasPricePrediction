from sklearn.model_selection import train_test_split
import pandas as pd
from data_scraper import *  # Import functions from data_scraper.py
from linear_Regression import *
from polinomial_Regression import *


# Define a function to load data
def load_data():
    try:
        # Get the filename from the user using the get_filename function
        file_name = get_filename()
        # Read data from a CSV file into a Pandas DataFrame
        data = pd.read_csv(file_name)
        # Check if a previous session's data is found
        print(f"{data} \nFound a previous session...")
        choice = input("Would you like to use this data? (y/n)\n> ")
        if choice == "y":
            return data  # Return the existing data
        else:
            data_scraper()  # If 'n', scrape new data
            return load_data()  # Recursively call load_data() to load the new data

    # Handle KeyboardInterrupt to exit gracefully
    except KeyboardInterrupt:
        print("Task interrupted successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print("Exception encountered:", e)
        print("Couldn't find data from a previous session. \n Generating New Data")
        data_scraper()  # Scrape new data
        return load_data()  # Recursively call load_data() to load the new data


def get_float(string):
    try:
        n = input(string)
        n = float(n)
        return n
    except Exception as e:
        print("Error! Wrong number: Expected Float", e)
        return get_float(string)


def get_int(string):
    try:
        n = input(string)
        n = int(n)
        return n
    except Exception as e:
        print("Error! Wrong number: Expected Integer", e)
        return get_int(string)


def forecaster():
    try:
        # Load data using the load_data function
        data = load_data()
        print(data)

        # Perform feature scaling on the data
        data = feature_scaling(data)
        print("Normalized Data:")
        print(data)

        # Split the data into training and testing sets using train_test_split
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        # Define learning rate and number of iterations for gradient descent
        learning_rate = 0.001
        num_iterations = 1000

        # Perform linear regression using gradient descent and get the slope (m) and intercept (q)
        m, q = gradient_descent(train_data, learning_rate, num_iterations)

        # Print the final slope and intercept
        print(f"Final slope (m): {m}")
        print(f"Final intercept (q): {q}")

        # Calculate and print the Mean Squared Error on the test set for linear regression
        mse_linear = calculate_linear_mse(test_data, m, q)
        print(f"Mean Squared Error on Test Set for the linear regression is: {mse_linear}")

        # Perform polynomial regression with a degree of 16 and get the coefficients
        degree = 16  # degree of the polynomial
        coefficients = polynomial_regression(train_data, degree)

        # Calculate and print the Mean Squared Error on the test set for polynomial regression
        mse_polynomial = calculate_polynomial_mse(test_data, coefficients)
        print(f"Mean Squared Error on Test Set for the polynomial regression is {mse_polynomial}")

        # Plot both linear and polynomial regression models along with the data
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
