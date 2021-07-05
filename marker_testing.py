import cv2
import cv2.aruco as aruco

camera = cv2.VideoCapture(1)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters_create()

aruco_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
aruco_params.minOtsuStdDev = 6.0

print(dir(aruco_params))

while True:
    _, img = camera.read()

    if img is not None:
        (corners, ids, rejected) = aruco.detectMarkers(
            img, aruco_dict, parameters=aruco_params)

        for i in range(0, len(rejected)):
            detected_marker = rejected[i]

            # Get the array of corners from the marker
            corners_unpacked = detected_marker[0]

            for j in range(0, len(corners_unpacked)):
                pt1 = tuple(map(int, corners_unpacked[j]))
                # Get the next point, wrapping around at the end.
                pt2 = tuple(map(int,
                                corners_unpacked[
                                    (j + 1) % len(corners_unpacked)]))

                cv2.line(img, pt1, pt2, (0, 0, 255), 2)

        for i in range(0, len(corners)):
            detected_marker = corners[i]

            # Get the array of corners from the marker
            corners_unpacked = detected_marker[0]

            for j in range(0, len(corners_unpacked)):
                pt1 = tuple(map(int, corners_unpacked[j]))
                # Get the next point, wrapping around at the end.
                pt2 = tuple(map(int,
                                corners_unpacked[
                                    (j + 1) % len(corners_unpacked)]))

                cv2.line(img, pt1, pt2, (255, 255, 0), 2)

        cv2.imshow("Image", img)

    if cv2.waitKey(1) == 27:
        break  # esc to quit
