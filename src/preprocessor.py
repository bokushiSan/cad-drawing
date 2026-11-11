import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Преобразование изображения чертежа из BGR в оттенки серого.

    Args:
        image: Изображение в виде BGR.

    Returns:
        Изображение в оттенках серого.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale


def binarize(image_gray: np.ndarray) -> np.ndarray:
    """
    Преобразование изображения из оттенков серого в бинарное методом Оцу

    Args:
        image_gray: Изображение в оттенках серого.

    Returns:
        Бинарное изображение.
    """
    threshold_val, binary_image = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold_val, binary_image