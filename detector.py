import collections
from pathlib import Path
from typing import Iterable

import cv2 as cv
import numpy as np


def get_classifiers_paths(classdir: Path) -> Iterable[Path]:
    # Zkontroluj, jestli `classdir` odkazuje na platný adresář
    if not isinstance(classdir, Path) or not Path.is_dir(classdir):
        raise NotADirectoryError(f'`{classdir}` is not a directory.')

    # Vrať tuple souborů v `classdir`
    return Path.iterdir(classdir)


def get_classifiers(classdir: Path) -> Iterable[cv.CascadeClassifier]:
    # Získej cestu každého klasifikátoru
    paths = get_classifiers_paths(classdir)
    classifiers = []

    # Přidej klasifikátor do `clasaifiers` jestli je platným klasifikátorem OpenCV
    for path in paths:
        if path.name == '.gitkeep':
            continue

        try:
            classifier = cv.CascadeClassifier(str(path))
            classifiers.append(classifier)
        except Exception as e:
            print(f'Error while loading classifier at {str(path)}: {str(e)}')

    # Vrať tuple klasifikátorů
    return tuple(classifiers)


def get_face_coordinates(img: np.ndarray,
                         classifiers: Iterable[cv.BaseCascadeClassifier or cv.CascadeClassifier]) -> np.ndarray:
    # Zkontroluj, jestli `img` je instance NumPy pole
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied `img` argument must be an instance of `numpy.ndarray`.')
    # Zkontroluj, jestli `classifiers` je procházitelný
    if not isinstance(classifiers, collections.Iterable):
        raise TypeError('Supplied `classifier` argument must be iterable.')

    img = np.copy(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = None

    # Projdi klasifikátory a přidej nalezené obliřeje do `faces`
    for classifier in classifiers:
        # Zkontroluj, jestli klasifikátor je instancí CV2 Classifiers
        if not isinstance(classifier, (cv.CascadeClassifier, cv.BaseCascadeClassifier)):
            raise TypeError('Supplied classifier is not an instance of cv.-Classifier.')

        # Najdi obličeje v obrázku
        found = classifier.detectMultiScale(img, 1.1, 5, cv.CASCADE_SCALE_IMAGE, (20, 20))
        # Přidej nalezené obličeje do `faces`
        if len(found) > 0:
            if faces is None:
                faces = found
            else:
                faces = np.concatenate((faces, found))

    return faces
