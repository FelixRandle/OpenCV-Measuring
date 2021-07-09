"""
Utility functions for assorted cv2 actions
"""

import cv2

from .config import CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_ID


def place_text(image, position, text, scale=2.0, color=(2, 255, 2)):
    cv2.putText(image, text, org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale,
                color=color, thickness=2,
                lineType=cv2.LINE_AA)


def register_params(params: dict) -> None:
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 1000, 480)

    for key, values in params.items():
        cv2.createTrackbar(key, "Parameters", values[0], values[1],
                           lambda _: ())


def get_param_value(key: str, default: float) -> float:
    try:
        return cv2.getTrackbarPos(key, "Parameters")
    except:
        return default


def get_camera(camera_id: int = CAMERA_ID):
    """
    :param camera_id: ID of the camera to retrieve, defaults to CAMERA_ID
    from utils/config.py
    :return: Camera Object
    """
    camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if camera is None or not camera.isOpened():
        return False
    else:
        print(
            f"CV_CAP_PROP_FRAME_WIDTH: '#"
            f"{camera.get(cv2.CAP_PROP_FRAME_WIDTH)}'")
        print(
            f"CV_CAP_PROP_FRAME_HEIGHT : '#"
            f"{camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}'")
        print(f"CAP_PROP_FPS : '#{camera.get(cv2.CAP_PROP_FPS)}'")
        print(f"CAP_PROP_POS_MSEC : '#{camera.get(cv2.CAP_PROP_POS_MSEC)}'")
        print(
            f"CAP_PROP_FRAME_COUNT  : '#"
            f"{camera.get(cv2.CAP_PROP_FRAME_COUNT)}'")

    return camera
