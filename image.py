from pathlib import Path
from typing import Tuple

import cv2 as cv
import numpy as np

import helper


def resize_by_width(img: np.ndarray, width: int) -> np.ndarray:
    # Check if `img` is an instance of NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')

    return cv.resize(img, (
        width,
        int(img.shape[0] / (img.shape[1] / width))
    ))


def get_by_dir(imgdir: Path) -> Tuple[Tuple[Path, np.ndarray]]:
    # Check if `imgdir` is a valid directory
    if not isinstance(imgdir, Path) or not Path.is_dir(imgdir):
        raise NotADirectoryError(f'`{imgdir}` is not a directory.')

    images = []

    # Iterate over possible images
    for file in Path.iterdir(imgdir):
        if Path.is_file(file):
            img = cv.imread(str(file))

            if img is not None and isinstance(img, np.ndarray):
                images.append((file, img))

    return tuple(images)


def crop_multi(img: np.ndarray, faces: np.ndarray, margin: float = 0) -> Tuple[np.ndarray]:
    # Check if `faces` and `img` are instances of NumPy array
    if not isinstance(faces, np.ndarray):
        raise TypeError('Supplied argument `faces` must be an instance of numpy.ndarray.')
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')

    images = []

    for (x, y, w, h) in faces:
        # Calculate border points with margin
        (p1, p2) = helper.calculate_borderpoints(img, (x, y, w, h), margin)

        # Add cropped image into `images`
        images.append(img[p1[1]:p2[1], p1[0]:p2[0]])

    return tuple(images)


def mark_faces(img: np.ndarray, faces: np.ndarray, margin: float = 0) -> np.ndarray:
    # Check if `img` and `faces` are NumPy arrays
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')
    if not isinstance(faces, np.ndarray):
        raise TypeError('Supplied argument `faces` must be an instance of numpy.ndarray.')

    img = np.copy(img)

    # Iterate through every found face and mark it in the picture
    for (x, y, w, h) in faces:
        # Calculate border points with margin
        (p1, p2) = helper.calculate_borderpoints(img, (x, y, w, h), margin)

        # Draw the rectangle marking the face
        cv.rectangle(img, p1, p2, (0, 0, 255), 2)

    return img
