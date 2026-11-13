import cv2
import numpy as np


def calculate_distance(binary_image: np.ndarray) -> np.ndarray:
    """
    Вычисление дистанции от пикселей линии до фона чертежа.

    Args:
        binary_image: Бинарное изображение чертежа.

    Returns:
        Изображение с расстоянием от пикселей линии до фона чертежа.
    """
    inverted_image = cv2.bitwise_not(binary_image)

    distance = cv2.distanceTransform(
        inverted_image,
        cv2.DIST_L2,  # евклидово расстояние
        5  # маска (5х5)
        )
    return distance

def calculate_skeleton(binary_image: np.ndarray) -> np.ndarray:
    """
    Вычисление скелета линии чертежа.

    Args:
        binary_image: Бинарное изображение чертежа.

    Returns:
        Изображение со скелетами линии чертежа.
    """
    inverted_image = cv2.bitwise_not(binary_image)
    skeleton = cv2.ximgproc.thinning(inverted_image)
    return skeleton
