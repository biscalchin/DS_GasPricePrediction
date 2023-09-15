# Import the necessary libraries
import numpy as np
import yfinance as yf  # For Yahoo Finance data retrieval
import pandas as pd    # For working with data in DataFrame format
import datetime as dt  # For working with date and time
from time import *     # For managing time-related operations
import warnings        # For silencing future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Defining the resource to monitor - in this case, natural gas futures (NG=F)
resLabel = "NG=F"
# Define the object 'NG' using the Ticker method from the Yahoo Finance API
NG = yf.Ticker(resLabel)


# Define the function to create the name of the CSV file in the format: {prefix}_NG_{today}{suffix}.csv
def get_filename(prefix="", suffix=""):
    today = dt.date.today()  # Get the current date
    return f"{prefix}_NG_{today}{suffix}.csv"  # Return the formatted filename


# Define the function to write a pandas DataFrame to a CSV file with a specified filename
def save_to_csv(data, prefix="", suffix=""):
    data.to_csv(get_filename(prefix, suffix), index=False)


# Define a function to get the period from the user with strict input validation
def get_period():
    while True:
        period = input(f"Choose the period in days:\n1. 7d\n2. 30d\n3. 60d\n>")
        if period == "1" or period == "7d" or period == "7":
            return "7d"
        elif period == "2" or period == "30d" or period == "30":
            return "30d"
        elif period == "3" or period == "60d" or period == "60":
            return "60d"
        else:
            print(f"Error: Invalid Choice!\n\n")


# Define a function to get the data retrieval interval from the user with input validation
def get_interval():
    while True:
        interval = input(f"Choose the interval:\n1. 1m\n2. 5m\n3. 15m\n>")
        if interval == "1" or interval == "1m" or interval == "1min":
            return "1m"
        elif interval == "2" or interval == "5min" or interval == "5m" or interval == "5":
            return "5m"
        elif interval == "3" or interval == "15min" or interval == "15m" or interval == "15":
            return "15m"
        else:
            print(f"Error: Invalid Choice!\n\n")


# Define the main listener function
def data_scraper():
    try:
        # Get user input for period and interval
        period = get_period()
        interval = get_interval()
        # Retrieve and save historical data based on user input
        data = NG.history(period=period, interval=interval)
        # launching the data cleaner
        data = data_cleaner(data)
        print("Data cleaning terminated. \n*** Data are now clean ***")
        sleep(0.8)
        print(data)
        sleep(0.8)
        prefix = f"{interval}_interval"
        suffix = f"_{period}_period"
        print(f"Saving data to {get_filename(prefix, suffix)}...")
        save_to_csv(data, prefix, suffix)
        print("Operation Succeed!")

    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")
    except Exception as e:
        print("Exception encountered:", e)


# Define a function called data_cleaner that takes a parameter called resource
def data_cleaner(resource):
    try:
        print("Cleaning...")
        print("Dropping useless columns...")
        # Drop the columns named Dividends and Stock Splits from resource along the horizontal axis
        resource = resource.drop(["Dividends", "Stock Splits"], axis=1)
        print("Operation Succeed!")
        print("Splitting indexes...")
        # Assign the index of resource to date_time and convert it to a series object
        date_time = resource.index
        date_time = date_time.to_series()
        print("Creating date and time column...")
        # Add two new columns to resource named Date and Hour, with values extracted from date_time
        resource["Date"] = date_time.dt.date
        resource["Hour"] = date_time.dt.time
        print("Operation Succeed!")
        # Return resource as the output of the function
        for i in range(len(resource)):
            resource.loc[:, 'Open'] = round(resource['Open'], 4)
            resource.loc[:, 'High'] = round(resource['High'], 4)
            resource.loc[:, 'Low'] = round(resource['Low'], 4)
            resource.loc[:, 'Close'] = round(resource['Close'], 4)
        return resource
    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")
    # Handle any other exception and print its message
    except Exception as e:
        print("Exception encountered:", e)


# Entry point of the script
if __name__ == '__main__':
    data_scraper()  # Call the main data_scraper function to start data retrieval and processing
