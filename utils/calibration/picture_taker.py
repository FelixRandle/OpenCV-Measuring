"""
Picture taker utility for calibrating the camera to remove distortion
"""
import sys

import cv2

from ..config import CAMERA_WIDTH
from ..cv2_common import get_camera
from ..image import scale_image


def take_pictures(image_path: str = 'images') -> None:
    camera = get_camera()
    if camera is False:
        print("Cannot initialise camera, exiting...", file=sys.stderr)
        sys.exit()

    print("CAMERA CALIBRATION:\n"
          "PRESS 's' to save a picture\n"
          "PRESS the escape key to continue")

    image_count = 0
    while True:
        _, img = camera.read()
        if img is not None:
            cv2.imshow("Live View", scale_image(img, 720 / CAMERA_WIDTH))

        res = cv2.waitKey(1)
        if res == ord('s'):
            cv2.imwrite(f'{image_path}/calibration_image_{image_count}.jpg',
                        img)
            image_count += 1
        elif res == 27:
            break  # esc to quit

    camera.release()
    cv2.destroyAllWindows()
