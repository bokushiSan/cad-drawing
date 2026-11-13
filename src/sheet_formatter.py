import numpy as np
from typing import Tuple


SHEET_FORMATS = {
    # "A0": (841, 1189),  # Временно убираю все кроме А3, А4, А4х3 (то, что есть в наборе)
    # "A1": (594, 841),
    # "A2": (420, 594),
    "A3": (420, 297),  # Пока он горизонтальный, иначе ValueError
    "A4": (210, 297),
    # "A5": (148, 210),
    "A4x3": (630, 297),
    }  # Можно дополнить, но пока этого хватит


def get_sheet_format(image: np.ndarray, tol: float = 0.02) -> Tuple[str, int, int]:
    """
    Определение формата листа по пропорциям.

    Args:
        image: 

    Returns:
        - Наименование формата
        - Ширина, мм
        - Высота, мм
    """
    height, width = image.shape[:2]
    ratio = width / height

    def ratio_distance(format_name: str) -> float:
        """
        Расчет отклонений пропорций изображения от заданного формата.
        """
        format_width, format_height = SHEET_FORMATS[format_name]
        format_ratio = format_width / format_height
        return abs(ratio / format_ratio - 1)
    
    best_format = min(SHEET_FORMATS, key=ratio_distance)
    error = ratio_distance(best_format)

    if error > tol:
        raise ValueError(f"Неизвестный формат листа: {width}x{height}")

    width_mm, height_mm = SHEET_FORMATS[best_format]

    return best_format, width_mm, height_mm


if __name__ == "__main__":
    for h, w in [
        (3521, 2493),
        (3508, 2481),
        (3521, 4973),
        (3520, 4973),
        (3307, 7000),
    ]:
        image = np.zeros((h, w), dtype=np.uint8)

        result = get_sheet_format(image)

        print(f"{w}x{h}: {result}")