import cv2
import numpy as np
from typing import Tuple


def get_features(component: np.ndarray) -> np.ndarray:
    """
    Получение координат, длин и углов наклона отрезка методом преобразования Хафа.

    Args:
        component: Маска компоненты.

    Returns:
        - x1, 
        - y1,
        - x2,
        - y2
        - Длина
        - Угол
    """
    lines = cv2.HoughLinesP(
        component,
        rho=1,  # Шаг, пикселей
        theta=np.pi/180,  # Шаг угла, 1 град.
        threshold=20,  # Минимальное число голосов в преобразовании Хафа
        minLineLength=20,  # Минимальное длина отрезка
        maxLineGap=5  # Максимальный разрв между линиями
    )
    if lines is None:
        return np.empty((0, 6), dtype=np.int32)
    
    lx = lines[:, 2] - lines[:, 0]
    ly = lines[:, 3] - lines[:, 1]

    l = np.hypot(lx, ly)
    angle = np.degrees(np.arctan2(ly, lx))
    angle = angle + 90 % 180 - 90
    return np.column_stack((lines, l, angle))

def is_roughness(component: np.ndarray) -> bool:
    """
    Определение по геометрическим характеристикам является ли компонента знаком шероховатости.

    Args:
        component: Маска компоненты.
    """
    height, width = component.shape

    features = get_features(component)

    if width / height < 3:  # Проверка, по которой знак шероховатости должен быть всегда вытянутым по горизонтали,
                            # тем самым отсекаем компоненты иных форм. 3 взято примерно, по ГОСТ 2.309-73
        return False

    downward_line = False
    upward_line = False
    horizontal_line = False

    for x1, y1, x2, y2, l, angle in features:
        if (abs(angle) <= 5) and (l >= 0.6 * width) and (max(y1, y2) <= 0.1 * height):  # Проверка горизонтальной
            horizontal_line = True
        if (abs(angle - 60) <= 5) and (l >= 0.5 * height):  # Проверка первой линии (на ЮВ)
            downward_line = True
        if (abs(angle + 60) <= 5) and (l >= 0.5 * height):  # Проверка второй линии (на СВ)
            upward_line = True

    return all([horizontal_line, downward_line, upward_line])


def remove_roughness_symbol(filtered: np.ndarray, labels: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """
    Удаление символа шероховатости с чертежа.

    Args:
        filtered: Маска после фильтрации.
        labels: Массив с указанием порядкового номера компоненты.
        stats: Массив характеристик каждой компоненты (X, Y, ширина, высота, кол-во пикселей).

    """
    image = filtered.copy()
    image_h, image_w = labels.shape

    all_component_labels = np.unique(labels[filtered > 0])  # Из пикселей в лейблы 

    for component_id in all_component_labels:
        x, y, w, h, px = stats[component_id]

        if (x < 0.5 * image_w) or (y + h > 0.5 * image_h):  # Должен быть в правом верхнем квадранте
            continue

        component_labels = labels[y:y+h, x:x+w]
        component = np.zeros(component_labels.shape, dtype=np.uint8)
        component[component_labels == component_id] = 255

        if is_roughness(component):
            image[labels == component_id] = 0

    return image

