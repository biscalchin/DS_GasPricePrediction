# Import the necessary libraries
import yfinance as yf  # For Yahoo Finance data retrieval
import datetime as dt  # For working with date and time
from progress_bar import *
import warnings        # For silencing future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# Defining the resource to monitor - in this case, natural gas futures (NG=F)
resLabel = "NG=F"
# Define the object 'NG' using the Ticker method from the Yahoo Finance API
NG = yf.Ticker(resLabel)


# Define the function to create the name of the CSV file in the format: {prefix}_NG_{today}{suffix}.csv
def get_filename():
    today = dt.date.today()  # Get the current date
    return f"NG_{today}.csv"  # Return the formatted filename


# Define the function to write a pandas DataFrame to a CSV file with a specified filename
def save_to_csv(data, filename):
    data.to_csv(filename, index=False)


# Define a function to get the period from the user with strict input validation
def get_period(interval="1m"):
    if interval == "1" or interval == "1m":
        return "7d"
    elif interval == "5m" or interval == "5":
        return "60d"
    else:
        print(f"Unknown Error occurred!\n\n")


# Define a function to get the data retrieval interval from the user with input validation
def get_interval():
    while True:
        interval = input(f"Choose the interval:\n1. 1m\n2. 5m\n>")
        if interval == "1" or interval == "1m" or interval == "1min":
            return "1m"
        elif interval == "2" or interval == "5min" or interval == "5m" or interval == "5":
            return "5m"
        else:
            print(f"Error: Invalid Choice!\n\n")


# Define the main listener function
def data_scraper():
    try:
        # Get user input for period and interval
        interval = get_interval()
        period = get_period(interval)
        # Retrieve and save historical data based on user input
        data = NG.history(period=period, interval=interval)
        # launching the data cleaner
        print("Data cleaning initiated. \n*** Data are now clean ***")

        data = data_cleaner(data)

        print(data)
        file_name = get_filename()
        print(f"Saving data to {file_name}...")

        save_to_csv(data, file_name)

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
        resource = resource.drop(["Dividends", "Stock Splits", "Low", "High", "Open"], axis=1)
        print("Operation Succeed!")
        # Return resource as the output of the function
        for i in range(len(resource)):
            resource.loc[:, 'Close'] = round(resource['Close'], 4)
        return resource
    # Handle KeyboardInterrupt to gracefully exit the program
    except KeyboardInterrupt:
        print("Task finished successfully")
    # Handle any other exception and print its message
    except Exception as e:
        print("Exception encountered:", e)

