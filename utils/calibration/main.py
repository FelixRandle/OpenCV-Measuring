def calibrate() -> None:
    import cv2
    import glob
    import numpy as np

    from .picture_taker import take_pictures

    # Make the user take pictures of the calibration chessboard.
    take_pictures()

    # Set our termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points
    # from all the calibration images.
    object_points = []
    img_points = []
    images = glob.glob('images/*.jpg')

    gray = None
    for file_name in images:
        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
        # If found, add object points, image points (after refining them)
        if ret:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)

    # Calibrate our camera using the object and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, img_points, gray.shape[::-1], None, None)

    # Save our variables to a file to use later.
    cv_file = cv2.FileStorage('calibration.yml', cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)

    cv_file.release()

