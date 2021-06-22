"""
Utility functions to clean up main file.
"""
import sys
import numpy as np

# Set debug if intelliJ debugger attached
DEBUG = False
if sys.gettrace() is not None:
    DEBUG = True

IS_WINDOWS = sys.platform == "win32"

# Can probably find a nicer way to see if there's a window to display on.
CAN_DISPLAY = IS_WINDOWS


def distance(pt1, pt2):
    length = (
        (pt2[0] - pt1[0]) ** 2 +
        (pt2[1] - pt1[1]) ** 2
    ) ** 0.5

    return length


def angle_between(pt1, pt2):
    ang1 = np.arctan2(*pt1[::-1])
    ang2 = np.arctan2(*pt2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def log(text):
    if not CAN_DISPLAY and DEBUG:
        print(text)
