from pathlib import Path
from typing import Tuple

import cv2 as cv
import numpy as np

import helper


def resize_by_width(img: np.ndarray, width: int) -> np.ndarray:
    # Zkontroluj, jestli `img` je instance NumPy pole
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')

    # Vrať obrázky se změněnou velikostí
    return cv.resize(img, (
        width,
        int(img.shape[0] / (img.shape[1] / width))
    ))


def get_by_dir(imgdir: Path) -> Tuple[Tuple[Path, np.ndarray]]:
    # Zkontroluj, jestli `imgdir` je platný adresář
    if not isinstance(imgdir, Path) or not Path.is_dir(imgdir):
        raise NotADirectoryError(f'`{imgdir}` is not a directory.')

    images = []

    # Projdi možné obrázky
    for file in Path.iterdir(imgdir):
        # Zkontroluj, jestli cesta je platná
        if Path.is_file(file):
            # Načti obrázek z cesty
            img = cv.imread(str(file))

            # Zkontroluj, jestli načtený soubor je obrazem a přidej ho do pole platných obrazů
            if img is not None and isinstance(img, np.ndarray):
                images.append((file, img))

    return tuple(images)


def crop_multi(img: np.ndarray, faces: np.ndarray, margin: float = 0) -> Tuple[np.ndarray]:
    # Zkontroluj, jestli `faces` a `img` jsou instance NumPy pole
    if not isinstance(faces, np.ndarray):
        raise TypeError('Supplied argument `faces` must be an instance of numpy.ndarray.')
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')

    images = []

    # Projdi obličeje
    for (x, y, w, h) in faces:
        # Spočítej hraniční body a přičti k nim okraj
        (p1, p2) = helper.calculate_borderpoints(img, (x, y, w, h), margin)

        # Přidej obrázek do `images`
        images.append(img[p1[1]:p2[1], p1[0]:p2[0]])

    return tuple(images)


def mark_faces(img: np.ndarray, faces: np.ndarray, margin: float = 0) -> np.ndarray:
    # Zkontroluj, jestli `faces` a `img` jsou instance NumPy pole
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')
    if not isinstance(faces, np.ndarray):
        raise TypeError('Supplied argument `faces` must be an instance of numpy.ndarray.')

    img = np.copy(img)

    # Projdi nalezené obličeje a označ je na obrázku
    for (x, y, w, h) in faces:
        # Spočítej hraniční body a přičti k nim okraj
        (p1, p2) = helper.calculate_borderpoints(img, (x, y, w, h), margin)

        # Nakresli okraj okolo nalezených obličejů
        cv.rectangle(img, p1, p2, (0, 0, 255), 2)

    return img
