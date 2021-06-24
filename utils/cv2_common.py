"""
Utility functions for assorted cv2 actions
"""

import cv2


def place_text(image, position, text, scale=2.0, color=(255, 255, 255)):
    cv2.putText(image, text, org=position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale,
                color=color, thickness=2,
                lineType=cv2.LINE_AA)


def register_params(params):
    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters", 1000, 480)

    for key, values in params.items():
        cv2.createTrackbar(key, "Parameters", values[0], values[1],
                           lambda _: ())


def get_param_value(key, default):
    try:
        return cv2.getTrackbarPos(key, "Parameters")
    except:
        return default


def get_camera(camera_id):
    camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 60)
    if camera is None or not camera.isOpened():
        return False
    else:
        print(
            f"CV_CAP_PROP_FRAME_WIDTH: '#{camera.get(cv2.CAP_PROP_FRAME_WIDTH)}'")
        print(
            f"CV_CAP_PROP_FRAME_HEIGHT : '#"
            f"{camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}'")
        print(f"CAP_PROP_FPS : '#{camera.get(cv2.CAP_PROP_FPS)}'")
        print(f"CAP_PROP_POS_MSEC : '#{camera.get(cv2.CAP_PROP_POS_MSEC)}'")
        print(f"CAP_PROP_FRAME_COUNT  : '#{camera.get(cv2.CAP_PROP_FRAME_COUNT)}'")

    return camera

