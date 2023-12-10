# Import the necessary libraries
import yfinance as yf  # For Yahoo Finance data retrieval
import datetime as dt  # For working with date and time
from progress_bar import *
import pandas as pd
import time
import warnings        # For silencing future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Defining the resource to monitor - in this case, natural gas futures (NG=F)
resLabel = "NG=F"
# Define the object 'NG' using the Ticker method from the Yahoo Finance API
NG = yf.Ticker(resLabel)


def get_filename():
    """
    Generates a filename for saving CSV data.

    The filename is in the format 'NG_{current_date}.csv', where {current_date} is today's date.

    Returns:
    - A string representing the generated filename.
    """
    today = dt.date.today()  # Get the current date
    return f"NG_{today}.csv"  # Return the formatted filename


def save_to_csv(data, filename):
    """
    Writes a pandas DataFrame to a CSV file.

    Parameters:
    - data: The pandas DataFrame to be written to the CSV.
    - filename: The name of the file to save the data to.
    """
    data.to_csv(filename, index=False)  # Save the DataFrame to a CSV file without including row indices


def get_period(interval="1m"):
    """
    Determines the data retrieval period based on the specified interval.

    Parameters:
    - interval: A string indicating the time interval (default is "1m").

    Returns:
    - A string representing the data retrieval period.
    """
    if interval == "1" or interval == "1m":
        return "7d"  # If the interval is 1 minute, return a period of 7 days
    elif interval == "5m" or interval == "5":
        return "60d"  # If the interval is 5 minutes, return a period of 60 days
    else:
        print(f"Unknown Error occurred!\n\n")


def get_interval():
    """
    Prompts the user to choose a data retrieval interval.

    Validates the user's input and only accepts specific values ("1m", "5m", etc.).

    Returns:
    - A string representing the user's choice of interval.
    """
    while True:
        interval = input(f"Choose the interval:\n1. 1m\n2. 5m\n>")
        if interval in ["1", "1m", "1min"]:
            return "1m"  # Return "1m" if the user selects a 1-minute interval
        elif interval in ["2", "5min", "5m", "5"]:
            return "5m"  # Return "5m" if the user selects a 5-minute interval
        else:
            print(f"Error: Invalid Choice!\n\n")


def data_scraper():
    """
    The main function to retrieve, clean, and save data.

    It handles user input for the interval and period, retrieves and cleans the data,
    and then saves it to a CSV file. It also handles exceptions and keyboard interrupts.
    """
    try:
        interval = get_interval()  # Get interval from user
        period = get_period(interval)  # Determine the period based on interval
        data = NG.history(period=period, interval=interval)  # Retrieve data
        print("Data cleaning initiated. \n*** Data are now clean ***")
        data = data_cleaner(data)  # Clean the retrieved data

        print(data)
        file_name = get_filename()  # Generate filename for saving
        print(f"Saving data to {file_name}...")
        save_to_csv(data, file_name)  # Save data to CSV

        print("Operation Succeed!")

    except KeyboardInterrupt:
        print("Task finished successfully")
    except Exception as e:
        print("Exception encountered:", e)


def data_cleaner(resource):
    """
    Cleans the provided data resource by dropping specific columns.

    Rounds the 'Close' values to 4 decimal places and drops unnecessary columns.

    Parameters:
    - resource: The pandas DataFrame to be cleaned.

    Returns:
    - The cleaned DataFrame.
    """
    try:
        print("Cleaning...")
        print("Dropping useless columns...")
        resource = resource.drop(["Dividends", "Stock Splits", "Low", "High", "Open"], axis=1)  # Drop specified columns
        print("Operation Succeed!")
        for i in range(len(resource)):
            resource.loc[:, 'Close'] = round(resource['Close'], 4)  # Round 'Close' values to 4 decimal places
        return resource
    except KeyboardInterrupt:
        print("Task finished successfully")
    except Exception as e:
        print("Exception encountered:", e)


def load_data(choice=""):
    """
    Loads data from a CSV file, offering the user the choice to use existing data or scrape new data.

    If existing data is found, the user is asked whether to use it or to scrape new data.

    Parameters:
    - choice: An optional string to bypass user input (useful for recursive calls).

    Returns:
    - A pandas DataFrame with the loaded data.
    """
    try:
        file_name = get_filename()  # Generate filename for loading
        data = pd.read_csv(file_name)  # Read data from the CSV file
        print(f"{data} \nFound data from a previous session.")
        if choice == "":
            choice = input("Would you like to use this data? (y/n)\n> ")

        if choice.lower() in ["y", "yes"]:
            return data  # Return existing data if user chooses so
        else:
            print("Collecting new data...")
            time.sleep(0.5)
            data_scraper()  # Scrape new data if user chooses so
            return load_data("y")  # Load the new data

    except KeyboardInterrupt:
        print("Task interrupted successfully")
    except Exception as e:
        print(f"Couldn't find data from a previous session.\nExtracting New Data")
        data_scraper()  # Scrape new data in case of exception
        return load_data("y")  # Load the new data


def get_float(string):
    """
    Prompts the user to enter a floating-point number and validates the input.

    If the input is not a valid float, the function recursively prompts again.

    Parameters:
    - string: The prompt string to be displayed to the user.

    Returns:
    - A floating-point number entered by the user.
    """
    try:
        n = input(string)
        return float(n)
    except Exception as e:
        print("Error! Wrong number: Expected Float", e)
        return get_float(string)  # Recursive call for re-prompting the user


def get_int(string):
    """
    Prompts the user to enter an integer and validates the input.

    If the input is not a valid integer, the function recursively prompts again.

    Parameters:
    - string: The prompt string to be displayed to the user.

    Returns:
    - An integer entered by the user.
    """
    try:
        n = input(string)
        return int(n)
    except Exception as e:
        print("Error! Wrong number: Expected Integer", e)
        return get_int(string)  # Recursive call for re-prompting the user
