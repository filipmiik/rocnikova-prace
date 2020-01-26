import collections
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np


def get_classifiers_paths(classdir: Path) -> Iterable[Path]:
    # Check if supplied path points to a directory
    if not isinstance(classdir, Path) or not Path.is_dir(classdir):
        raise NotADirectoryError(f'`{classdir}` is not a directory.')

    # Return a tuple of files in `classdir` path
    return Path.iterdir(classdir)


def get_classifiers(classdir: Path) -> Iterable[cv.CascadeClassifier]:
    # Get path of each classifier
    classifiers = get_classifiers_paths(classdir)

    # Create classifiers from paths
    return map(lambda path: cv.CascadeClassifier(str(path)), classifiers)


def get_face_coordinates(img: np.ndarray,
                         classifiers: Iterable[cv.BaseCascadeClassifier or cv.CascadeClassifier]) -> np.ndarray:
    # Check if `img` is a NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied `img` argument must be an instance of `numpy.ndarray`.')
    # Check if `classifiers` arg is iterable
    if not isinstance(classifiers, collections.Iterable):
        raise TypeError('Supplied `classifier` argument must be iterable.')

    img = np.copy(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = None

    # Iterate through classifiers and add matches to `faces` array
    for classifier in classifiers:
        # Check if supplied classifiers are indeed instances of CV2 Classifiers
        if not isinstance(classifier, (cv.CascadeClassifier, cv.BaseCascadeClassifier)):
            raise TypeError('Supplied classifier is not an instance of cv.-Classifier.')

        # Detect faces in image
        found = classifier.detectMultiScale(img, 1.1, 5, cv.CASCADE_SCALE_IMAGE, (20, 20))
        # Add found faces to `faces` array
        if len(found) > 0:
            if faces is None:
                faces = found
            else:
                faces = np.concatenate((faces, found))

    return faces
