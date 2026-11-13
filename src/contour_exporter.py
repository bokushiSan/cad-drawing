import json
import numpy as np
from pathlib import Path


def pixels_to_mm(segments: np.ndarray, image: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """
    Перевод из пикселей в миллиметры.
    """
    height_px, width_px = image.shape[:2]

    k_x = width_mm / width_px
    k_y = height_mm / height_px

    scale = np.array([k_x, k_y, k_x, k_y])

    segments_mm = segments * scale

    return np.round(segments_mm, decimals=3)

def save_contour(segments: np.ndarray, output_path: Path):
    """
    Сохранение координат отрезков в json.
    """
    contour = [
        {
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2) 
        } for x1, y1, x2, y2 in segments
    ]

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(contour, file, indent=4)
