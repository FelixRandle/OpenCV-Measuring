import cv2
import cv2.aruco as aruco

import numpy as np

import sys

from utils.common import DEBUG, CAN_DISPLAY, log, angle_between
from utils.cv2_common import get_camera, register_params, place_text,\
    get_param_value
from utils.image import scale_image, get_transformed_point, \
    four_point_transform, get_contours

# IntelliJ complains about this not existing but it works
from distances import HORIZONTAL, VERTICAL, MARKER_SIZE

camera = get_camera(1 if CAN_DISPLAY else 0)
if camera is None:
    print("Cannot initialise camera, exiting...", file=sys.stderr)

# Register params with format
# "NAME": [DEFAULT, MAX, FLOAT?]
if CAN_DISPLAY:
    register_params({
        "gaussian__ksize": [3, 20],
        "gaussian__sigmax": [150, 200],
        "canny__threshold1": [130, 255],
        "canny__threshold2": [40, 255],
        "erosion__kernel_size": [1, 20],
        "dilation__kernel_size": [3, 20],
        "border__epsilon": [2, 100]
    })

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters_create()

while True:
    return_value, img = camera.read()

    corner_positions = {
        0: None,
        1: None,
        2: None,
        3: None
    }

    if img is not None:
        (corners, ids, rejected) = aruco.detectMarkers(
            img, aruco_dict, parameters=aruco_params)

        if len(corners) < 4:
            log("Cannot see enough markers to do anything useful.")
            if CAN_DISPLAY:
                cv2.imshow("Image", img)
        elif len(corners) > 4:
            log("Too many markers in frame")
            if CAN_DISPLAY:
                cv2.imshow("Image", img)
        else:
            for i in range(0, len(corners)):
                detected_marker = corners[i]

                # Get the array of corners from the marker
                corners_unpacked = detected_marker[0]

                pt2 = None
                for j in range(0, len(corners_unpacked)):
                    pt1 = tuple(map(int, corners_unpacked[j]))
                    # Get the next point, wrapping around at the end.
                    pt2 = tuple(map(int,
                                    corners_unpacked[
                                        (j + 1) % len(corners_unpacked)]))

                corner_positions[ids[i][0]] = pt2

                log(f"Found marker ID: #{ids[i][0]} "
                    f"with top left at position #{pt2}")

            # Transform our image to get a birds eye view.
            img, M = four_point_transform(
                img,
                np.array(list(corner_positions.values()), np.float32))

            ###
            # PIXEL -> MM Calculation
            ###

            pixels_to_distance = {
                "vertical": [],
                "horizontal": []
            }

            (height, width, _) = img.shape
            pixels_to_distance["horizontal"].append(
                width / HORIZONTAL
            )

            pixels_to_distance["vertical"].append(
                height / VERTICAL
            )

            # Get size of all markers and use that for
            # our pixel->mm translation
            for i in range(0, len(corners)):
                corners_unpacked = corners[i][0]

                pt2 = None
                for j in range(0, len(corners_unpacked)):
                    pt1 = tuple(map(int, corners_unpacked[j]))
                    # Get the next point, wrapping around at the end.
                    pt2 = tuple(map(int,
                                    corners_unpacked[
                                        (j + 1) % len(corners_unpacked)]))

                    pt1 = get_transformed_point(pt1, M)
                    pt2 = get_transformed_point(pt2, M)

                    line_width = abs(pt1[0] - pt2[0])
                    line_height = abs(pt1[1] - pt2[1])

                    length = (
                                     line_width ** 2 +
                                     line_height ** 2
                             ) ** 0.5

                    ang = np.arctan(line_height / (line_width + 1e-25))

                    pixel_mm_ratio = length / MARKER_SIZE

                    if line_height != 0:
                        pixels_to_distance['vertical'].append(
                            line_height / (np.sin(ang) * MARKER_SIZE))

                    if line_width != 0:
                        pixels_to_distance['horizontal'].append(
                            line_width / (np.cos(ang) * MARKER_SIZE))

            average_horizontal_ratio = np.average(
                pixels_to_distance["horizontal"])
            average_vertical_ratio = np.average(
                pixels_to_distance["vertical"])

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

                    line_width = abs(pt1[0] - pt2[0])
                    line_height = abs(pt1[1] - pt2[1])

                    line_width = line_width / average_horizontal_ratio

                    line_height = line_height / average_vertical_ratio

                    real_distance = round((line_width ** 2 +
                                           line_height ** 2) ** 0.5, 1)

                    place_text(
                        img, text=f"{real_distance}mm",
                        position=(int((pt1[0] + pt2[0]) / 2),
                                  int((pt1[1] + pt2[1]) / 2)),
                        scale=0.5)

            if CAN_DISPLAY:
                print(f"display: {CAN_DISPLAY}")
                if DEBUG:
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

                else:
                    cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break  # esc to quit

camera.release()
cv2.destroyAllWindows()
