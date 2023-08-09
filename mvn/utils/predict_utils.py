import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(self, video_paths, intrinsic_params, extrinsic_params):
        self.video_paths = video_paths
        self.proj_matrices = np.matmul(intrinsic_params, extrinsic_params)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        # Read video and extract frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            frames.append(frame)
        cap.release()

        frames = torch.stack(frames)

        P = self.projection_matrix
        proj_matrices = torch.tensor(self.proj_matrix, dtype=torch.float32).repeat(
            len(frames), 1, 1
        )

        return frames, proj_matrices


def calibrate_camera(images, pattern_size, square_size):
    """
    :param images: List of images containing the calibration pattern.
    :param pattern_size: The number of inner corners in the pattern. (Width x Height)
    :param square_size: The size of one square in the pattern, typically in centimeters.
    :return: Camera matrix (intrinsic parameters) and distortion coefficients.
    """
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    return ret, mtx, dist

def compute_extrinsics(image, mtx, dist, pattern_size=(7, 6), square_size=2.5):
    objp = np.zeros((pattern_size[1] * pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners, mtx, dist)
        return rvecs, tvecs
    else:
        return None, None


def get_parameters(mtx, rvecs, tvecs):
    # Intrinsic Parameters
    intrinsic_params = mtx

    # Convert rvecs (rotation vector) to a rotation matrix
    R, _ = cv2.Rodrigues(rvecs)

    # Extrinsic Parameters
    extrinsic_params = np.hstack((R, tvecs))

    return intrinsic_params, extrinsic_params

def params_from_images(calibration_video_path, pattern_size=(7, 6), square_size=2.5):
    cap = cv2.VideoCapture(calibration_video_path)
    images = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and normalize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        images.append(frame)
    cap.release()

    _, mtx, dist = calibrate_camera(images, pattern_size, square_size)
    rvecs, tvecs = compute_extrinsics(images[0], mtx, dist, pattern_size, square_size)
    return get_parameters(mtx, rvecs, tvecs)