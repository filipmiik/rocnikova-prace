from typing import Tuple, Iterable

import numpy as np
import collections


def calculate_borderpoints(img: np.ndarray, coords: Iterable[int], margin: float or int = 0) \
        -> Tuple[Tuple[int, int], Tuple[int, int]]:
    # Check if `img` is an instance of NumPy array
    if not isinstance(img, np.ndarray):
        raise TypeError('Supplied argument `img` must be an instance of numpy.ndarray.')
    # Check if `coords` is iterable
    if not isinstance(coords, collections.Iterable):
        raise TypeError('Supplied argument `coords` must be iterable.')
    # Check if `margin` is of type float or int
    if not isinstance(margin, (int, float)):
        raise TypeError('Supplied argument `margin` must be of type `int` or `float`.')

    (x, y, w, h) = coords
    xmar = int(w * margin)
    ymar = int(h * margin)

    p1 = (
        int(0 if x - xmar < 0 else x - xmar),
        int(0 if y - ymar < 0 else y - ymar)
    )
    p2 = (
        int(0 if x + w + xmar > img.shape[1] else x + w + xmar),
        int(0 if y + h + ymar > img.shape[0] else y + h + ymar)
    )

    return p1, p2
