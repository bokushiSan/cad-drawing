import cv2
import numpy as np
from typing import Tuple


def get_components(contours: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Получение связанных непрерывных компонент и их статистики.

    Args:
        contours: Изображение маски с контурами.
    
    Returns:
        - Количество связанных компонент.
        - Массив с указанием порядкового номера компоненты.
        - Массив характеристик каждой компоненты (X, Y, ширина, высота, кол-во пикселей).
        - Координаты центров масс компонент.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(contours, connectivity=8)
    return num_labels, labels, stats, centroids


def filter_components(
        labels: np.ndarray, 
        stats: np.ndarray, 
        min_area: int = 500, 
        max_rel_size: float = 0.9
    ) -> np.ndarray:
    """
    Фильтрация связанных компонент по количеству пикселей.

    Args:
        labels: Массив с указанием порядкового номера компоненты.
        stats: Массив характеристик каждой компоненты (X, Y, ширина, высота, кол-во пикселей).
        min_area: Минимальное количество пикселей компоненты (эмпирически 500).
        max_rel_size: Порог для исключения компонентов относительно размера изображения (эмпирически 0.9), д.ед.

    Returns:
        Отфильтрованное изображение.
    """
    image_height, image_width = labels.shape

    areas = stats[:, 4]  # Колво пикселей
    widths = stats[:, 2]  # Ширина
    heights = stats[:, 3]  # Высота

    rel_widths = widths / image_width
    rel_heights = heights / image_height

    valid_area = areas >= min_area  # Фильтр мелких частей

    oversized = ((rel_widths >= max_rel_size) & (rel_heights >= max_rel_size))  # Фильтр рамки

    valid_labels = valid_area & ~oversized
    valid_labels[0] = False  # Фильтр фона

    filtered = np.zeros(labels.shape, dtype=np.uint8)
    filtered[valid_labels[labels]] = 255

    return filtered
