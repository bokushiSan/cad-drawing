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

def get_thickness(distance_image: np.ndarray, skeleton_image: np.ndarray) -> np.ndarray:
    """
    Получение толщины линии чертежа.

    Args:
        distance_image: Изображение с расстоянием от пикселей линии до фона чертежа.
        skeleton_image: Изображение со скелетами линии чертежа.

    Returns:
        Толщина линий, пиксели.
    """
    thickness = 2 * distance_image[skeleton_image > 0]
    return thickness

def build_overlay_thickness(image: np.ndarray, distance_image: np.ndarray, skeleton_image: np.ndarray) -> np.ndarray:
    """
    Получение изображения с цветной визуализацией линий по скелету.

    Args:
        image: Исходное изображение чертежа.
        distance_image: Изображение с расстоянием от пикселей линии до фона чертежа.
        skeleton_image: Изображение со скелетами линии чертежа.

    Returns:
        Изображение с цветной визуализацией линий по скелету.
    """
    overlay = image.copy()
    thickness_map = 2 * distance_image  # толщина только в скелете
    ys, xs = np.where(skeleton_image > 0)
    for y, x in zip(ys, xs):
        th = thickness_map[y, x]
        # эвристически подобраные интервалы
        if th > 8:
            color = (0, 0, 255)
        elif th < 4:  
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        overlay[y, x] = color

    return overlay
