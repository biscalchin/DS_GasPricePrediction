# Import necessary libraries and modules
import matplotlib.pyplot as plt  # For plotting data
import numpy as np  # For numerical operations
import pandas as pd  # For data handling
from LinearRegression import LinearRegression  # Custom LinearRegression class
from sklearn.model_selection import train_test_split  # For splitting data
from data_scraper import *  # Import functions from data_scraper.py


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


# Define a function to calculate mean squared error
def mean_sq_err(y_test, prediction):
    return np.mean((y_test - prediction) ** 2)


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


# Define the main application function
def app():
    try:
        # Load data using the load_data function
        dataset = load_data()
        print(dataset)

        # Prepare the input features (X) and target variable (y)
        X = dataset.index.values.reshape(-1, 1)  # Reshape to make it a 2D array
        y = dataset["Close"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

        # getting the lr and the amount of the iterations
        learning_rate = get_float("Choose a learning rate: \n>")
        num_iteration = get_int("Choose the amount of the iterations: \n>")

        # Create an instance of your custom LinearRegression class
        reg = LinearRegression(lr=learning_rate, n_iters=num_iteration)

        # Fit the linear regression model to the training data
        reg.fit(X_train, y_train)

        # Make predictions on the test data
        prediction = reg.predict(X_test)

        # Calculate the mean squared error between the actual and predicted values
        mse = mean_sq_err(y_test, prediction)
        print(mse)

        # Predict the target variable for all data points
        y_pred_line = reg.predict(X)

        # Create a scatter plot of the training and testing data points
        cmap = plt.get_cmap('viridis')
        fig = plt.figure(figsize=(8, 6))
        m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=1)
        m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=1)

        # Plot the regression line
        plt.plot(X, y_pred_line, color="Black", linewidth=1, label='Prediction')

        # Display the plot
        plt.show()

    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")

    # Handle other exceptions and print their messages
    except Exception as e:
        print("Exception encountered:", e)


# Entry point of the script
if __name__ == '__main__':
    app()  # Call the main application function when the script is executed
