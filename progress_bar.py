import colorama
import time
import threading
import sys
import itertools


def progress_bar(progress, total, color=colorama.Fore.YELLOW):
    """
    Displays a progress bar in the console.

    Parameters:
    - progress (int): The current progress of the task.
    - total (int): The total value corresponding to 100% completion.
    - color (colorama.Fore): The initial color of the progress bar (default is yellow).

    The color of the progress bar changes based on the completion percentage:
    - Red for less than 33% completion.
    - Yellow for 33% to 99% completion.
    - Green for 100% completion.
    """
    # Define color codes
    green = colorama.Fore.GREEN
    red = colorama.Fore.RED
    yellow = colorama.Fore.YELLOW
    reset = colorama.Fore.RESET

    # Calculate the percentage of completion
    percent = 100 * (progress / float(total))
    # Create the progress bar string
    bar = '█' * int(percent) + '-' * (100 - int(percent))

    # Change color based on completion percentage
    if progress < (total / 3):
        color = red
    elif (total / 3) <= progress < total:
        color = yellow
    elif progress == total:
        color = green

    # Print the progress bar
    print(color + f"\r|{bar}| {percent:.2f}%" + reset, end="\r")


class Spinner:
    """
    A simple console spinner animation.

    Uses threading to run the spinner animation in the background.

    Attributes:
    - spinner: An iterator cycling through spinner characters.
    - delay (float): The delay in seconds between spinner updates.
    - stop_running (threading.Event): Event to stop the spinner animation.
    - spin_thread (threading.Thread): The thread running the spinner animation.
    """

    def __init__(self, delay=0.1):
        """
        Initializes the spinner.

        Parameters:
        - delay (float): The delay in seconds between spinner updates.
        """
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])  # Spinner character sequence
        self.delay = delay  # Delay between spinner updates
        self.stop_running = threading.Event()  # Event to control stopping
        self.spin_thread = threading.Thread(target=self.init_spin)  # Spinner thread

    def init_spin(self):
        """
        The method run by the spinner thread.

        Continuously updates the spinner animation until stop_running is set.
        """
        while not self.stop_running.is_set():
            sys.stdout.write(next(self.spinner))  # Write the next spinner character
            sys.stdout.flush()  # Ensure the character is displayed
            time.sleep(self.delay)  # Wait for the specified delay
            sys.stdout.write('\b')  # Backspace to erase the last character

    def start(self):
        """
        Starts the spinner animation.

        This method begins the thread that runs the spinner.
        """
        self.spin_thread.start()  # Start the spinner thread

    def stop(self):
        """
        Stops the spinner animation.

        Sets the stop_running event and waits for the thread to finish.
        """
        self.stop_running.set()  # Signal to stop the spinner
        self.spin_thread.join()  # Wait for the spinner thread to finish
        sys.stdout.write('\b')  # Erase the last spinner character
