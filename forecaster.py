from sklearn.model_selection import train_test_split
import pandas as pd
from data_scraper import *  # Import functions from data_scraper.py
from linear_Regression import *
from polinomial_Regression import *
import time


def calculate_accuracy(mse):
    try:
        accuracy = (1 - mse) * 100  # as per your script
        return round(accuracy, 2)
    except Exception as e:
        print("Error in calculating accuracy: ", e)
        return None


# Define a function to load data
def load_data(choise=""):
    try:
        # Get the filename from the user using the get_filename function
        file_name = get_filename()
        # Read data from a CSV file into a Pandas DataFrame
        data = pd.read_csv(file_name)
        # Check if a previous session's data is found
        print(f"{data} \nFound data from a previous session.")
        if choise == "":
            choise = input("Would you like to use this data? (y/n)\n> ")

        if choise == "y" or choise == "yes" or choise == "Y":
            return data  # Return the existing data
        else:
            print("Collecting new datas...")
            time.sleep(0.5)
            data_scraper()  # If 'n', scrape new data
            return load_data("y")  # Recursively call load_data() to load the new data

    # Handle KeyboardInterrupt to exit gracefully
    except KeyboardInterrupt:
        print("Task interrupted successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print(f"Couldn't find data from a previous session.\n Extracting New Data")
        data_scraper()  # Scrape new data
        return load_data("y")  # Recursively call load_data() to load the new data


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

        # Plot both linear and polynomial regression models along with the data
        print("Plotting Linear and Polynomial Regression Models along with the Data...")

        # For Linear Regression
        accuracy_linear = calculate_accuracy(mse_linear)
        print(f"Linear regression prediction accuracy: {accuracy_linear}%")

        # For Polynomial Regression
        accuracy_polynomial = calculate_accuracy(mse_polynomial)
        print(f"Polynomial regression prediction accuracy: {accuracy_polynomial}%")

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
