import cv2
import cv2.aruco as aruco

import numpy as np

import sys

from utils.common import DEBUG, CAN_DISPLAY, log, angle_between
from utils.cv2_common import get_camera, register_params, place_text, \
    get_param_value
from utils.image import scale_image, get_transformed_point, \
    four_point_transform, get_contours

# IntelliJ complains about this not existing but it works
from distances import HORIZONTAL, VERTICAL, MARKER_SIZE

camera = get_camera(2 if CAN_DISPLAY else 0)
if camera is None:
    print("Cannot initialise camera, exiting...", file=sys.stderr)

# Register params with format
# "NAME": [DEFAULT, MAX, FLOAT?]
if CAN_DISPLAY:
    register_params({
        "gaussian__ksize": [5, 20],
        "gaussian__sigmax": [172, 200],
        "canny__threshold1": [130, 255],
        "canny__threshold2": [40, 255],
        "canny__aperture": [1, 3],
        "erosion__kernel_size": [1, 20],
        "dilation__kernel_size": [2, 20],
        "border__epsilon": [2, 100],
        "adthresh__block_size": [6, 15],
        "adthresh__C": [2, 10]
    })

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters_create()

while True:
    img = cv2.imread("test_image2.jpg")

    img = scale_image(img, 0.25)

    ###
    # Image analysis
    ###

    contours, (grey, blurred,
               edged, eroded, dilated) = get_contours(img)

    epsilon_multiplier = get_param_value("border__epsilon", 2) / 100
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)

        epsilon = epsilon_multiplier * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        for i in range(0, len(approx)):
            pt1 = approx[i][0]

            pt2 = approx[(i + 1) % len(approx)][0]

            cv2.line(img, pt1, pt2, (100, 0, 255), 2)

    row1 = np.hstack((
        scale_image(img),
        scale_image(
            cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)),
        scale_image(
            cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR))))

    row2 = np.hstack((
        scale_image(
            cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)),
        scale_image(
            cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)),
        scale_image(
            cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR))
    ))

    cv2.imshow("DEBUG", np.vstack((row1, row2)))

    if cv2.waitKey(1) == 27:
        break  # esc to quit

camera.release()
cv2.destroyAllWindows()
