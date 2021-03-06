"""
Utility functions to clean up main file.
"""
import argparse
import sys
import numpy as np
import cv2

# Set debug if intelliJ debugger attached
DEBUG = False
if sys.gettrace() is not None:
    DEBUG = True

IS_WINDOWS = sys.platform == "win32"

# Can probably find a nicer way to see if there's a window to display on.
parser = argparse.ArgumentParser(description='Process Arguments')
parser.add_argument('-c', '--console',
                    help='Output to console instead of screen',
                    action='store_true')

args = parser.parse_args()

CAN_DISPLAY = not args.console


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


def load_coefficients(path: str) -> [np.ndarray, np.ndarray]:
    """
    Load coefficients from camera calibration

    :param path: File to load coefficients from.
    :return: Camera Matrix and Dist Matrix
    """
    # FILE_STORAGE_READ
    coefficient_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = coefficient_file.getNode("K").mat()
    dist_matrix = coefficient_file.getNode("D").mat()

    coefficient_file.release()
    return [camera_matrix, dist_matrix]

