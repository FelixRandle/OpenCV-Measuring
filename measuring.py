import cv2
import cv2.aruco as aruco

import numpy as np

import sys

from utils.common import DEBUG, CAN_DISPLAY, log
from utils.cv2_common import get_camera, register_params, place_text
from utils.image import scale_image, get_transformed_point, \
    four_point_transform, get_contours

# IntelliJ complains about this not existing but it works
from distances import HORIZONTAL, VERTICAL, MARKER_SIZE

camera = get_camera(1 if CAN_DISPLAY else 0)
if camera is None:
    print("Cannot initialise camera, exiting...", file=sys.stderr)

# Register params with format
# "NAME": [DEFAULT, MAX, FLOAT?]
register_params({
    "gaussian__ksize": [3, 20],
    "gaussian__sigmax": [150, 200],
    "canny__threshold1": [130, 255],
    "canny__threshold2": [40, 255],
    "erosion__kernel_size": [1, 20],
    "dilation__kernel_size": [3, 20]
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

        blank_img = img.copy()

        if len(corners) < 4:
            log("Cannot see enough markers to do anything useful.")
            cv2.imshow("Image", img)
        elif len(corners) > 4:
            print("Too many markers in frame")
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

            if any(value is None for value in corner_positions.items()):
                print(corners)
                print(corner_positions)

            # Transform our image to get a birds eye view.
            img, M = four_point_transform(
                blank_img,
                np.array(list(corner_positions.values()), np.float32))

            ###
            # PIXEL -> MM Calculation
            ###

            pixels_to_distance = []

            (width, height, _) = img.shape

            pixels_to_distance.append(
                width / HORIZONTAL
            )

            pixels_to_distance.append(
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

                    length = (
                                     (pt1[0] - pt2[0]) ** 2 +
                                     (pt1[1] - pt2[1]) ** 2
                             ) ** 0.5

                    pixels_to_distance.append(
                        length / MARKER_SIZE
                    )

            # for key, value in corner_positions.items():
            #     # Translate our corner positions to the new, translated image
            #     print(f"Original: {value}")
            #     value = get_transformed_point(value, M)
            #     corner_positions[key] = value
            #     print(f"Translated: {value}")
            #
            #
            #     # Calculate pixel to mm ratio from our markers
            #     # TODO: take the size of our markers into account
            #     # when calculating this.
            #
            #     pt1 = value
            #     for i in range(key + 1, 4):
            #         pt2 = corner_positions[i]
            #         if pt2 is not None:
            #             expected_distance = KNOWN_DISTANCES[key][i].value
            #
            #             length = utils.distance(pt1, pt2)
            #
            #             log(f"Line with pixel length {length} "
            #                       f"and expected length {expected_distance} "
            #                       f"from {key} to {i} has pixel->mm of "
            #                       f"{length / expected_distance}")



            average_pixel_distance = np.average(pixels_to_distance)

            log(f"Average pixels per mm = {average_pixel_distance}")

            ###
            # Image analysis
            ###

            contours, \
            (grey, blurred, edged, eroded, dilated) = get_contours(img)

            # TODO: Optimise this more, quite laggy.
            for cnt in contours:
                perimeter = cv2.arcLength(cnt, True)

                epsilon = 0.02 * perimeter
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                for i in range(0, len(approx)):
                    pt1 = approx[i][0]

                    pt2 = approx[(i + 1) % len(approx)][0]

                    cv2.line(img, pt1, pt2, (100, 0, 255), 2)

                    line_length = ((abs(pt2[0] - pt1[0]) ** 2) +
                                   (abs(pt2[1] - pt1[1])) ** 2) ** 0.5

                    real_distance = round(
                        line_length / average_pixel_distance, 1)

                    log(f"line with pixel distance of {line_length} "
                              f"using average of "
                              f"{average_pixel_distance} has "
                              f"real value of {real_distance}")

                    place_text(
                        img, text=f"{real_distance}mm",
                        position=(int((pt1[0] + pt2[0]) / 2),
                                  int((pt1[1] + pt2[1]) / 2)),
                        scale=0.5)

            if CAN_DISPLAY:
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
