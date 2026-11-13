import cv2
import numpy as np


def get_contour(distance_image: np.ndarray, skeleton_image: np.ndarray, min_thickness: float = 4) -> np.ndarray:
    """
    Получение контура по заданной толщине линий.
    """
    thickness_map = 2 * distance_image  # Толщина только в скелете
    sk_mask = (skeleton_image > 0) & (thickness_map >= min_thickness)

    main_lines = np.zeros(skeleton_image.shape, dtype=np.uint8)
    main_lines[sk_mask] = 255

    return main_lines