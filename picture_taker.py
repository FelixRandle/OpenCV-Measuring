"""
Picture taker utility for camera calibration images
"""
import cv2
from utils.image import scale_image

camera = cv2.VideoCapture(2)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
camera.set(cv2.CAP_PROP_FPS, 30)

image_count = 0
while True:
    _, img = camera.read()

    cv2.imshow("image", scale_image(img, 0.4))

    res = cv2.waitKey(1)
    if res == ord('s'):
        cv2.imwrite(f'images/chessboard_{image_count}.jpg', img)
        image_count += 1
    elif res == 27:
        break  # esc to quit


camera.release()
cv2.destroyAllWindows()
