import colorama
import time
import threading
import sys
import itertools


def progress_bar(progress, total, color=colorama.Fore.YELLOW):
    green = colorama.Fore.GREEN
    red = colorama.Fore.RED
    yellow = colorama.Fore.YELLOW
    reset = colorama.Fore.RESET

    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))

    if progress < (total / 3):
        color = red
    elif (total / 3) <= progress < total:
        color = yellow
    elif progress == total:
        color = green

    print(color + f"\r|{bar}| {percent:.2f}%" + reset, end="\r")



class Spinner:
    def __init__(self, delay=0.1):
        self.spinner = itertools.cycle(['-', '/', '|', '\\'])
        self.delay = delay
        self.stop_running = threading.Event()
        self.spin_thread = threading.Thread(target=self.init_spin)

    def init_spin(self):
        while not self.stop_running.is_set():
            sys.stdout.write(next(self.spinner))  # write the next character
            sys.stdout.flush()
            time.sleep(self.delay)
            sys.stdout.write('\b')  # erase the last written char

    def start(self):
        self.spin_thread.start()

    def stop(self):
        self.stop_running.set()
        self.spin_thread.join()
        sys.stdout.write('\b')  # erase the spinner
