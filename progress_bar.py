import colorama


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
