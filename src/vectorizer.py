import cv2
import numpy as np
from typing import Tuple


def vectorize(skeleton: np.ndarray, eps: float = 2.0) -> np.ndarray:
    """
    Преобразование растрового скелета в отрезки.

    Args:
        skeleton: Бинарная маска контура.
        eps: Допуск при аппроксимации, пикс.
    Return:
        Массив координат отрезков (x1, y1, x2, y2).
    """
    contours, hierarchy = cv2.findContours(
        skeleton.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )  # Упорядоченные последовательности координат пикселей
    segments = set()

    for con in contours:
        points = cv2.approxPolyDP(con, eps, closed=True).reshape(-1, 2)

        if len(points) < 2:
            continue

        for idx, val in enumerate(points):
            p1 = tuple(map(int, val))
            p2 = tuple(map(int, points[(idx + 1) % len(points)]))

            if p1 == p2:
                continue

            start, end = sorted((p1, p2))
            segments.add((*start, *end))

    return np.array(sorted(segments), dtype=np.int32).reshape(-1, 4)
