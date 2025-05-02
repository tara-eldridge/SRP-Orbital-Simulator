# This file contains small functions to control code workflow
import sys

def pause():
    key = input("Press enter to continue or 'q' to quit.\n")
    if key == "q":
        sys.exit()
    else:
        pass