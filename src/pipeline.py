import numpy as np
import cv2
from pathlib import Path
from src.image_loader import load_image
from src.preprocessor import to_grayscale, binarize
from src.line_analyzer import calculate_distance, calculate_skeleton
from src.contour_extractor import get_contour
from src.component_analyzer import get_components, filter_components
from src.roughness_detector import remove_roughness_symbol
from src.vectorizer import vectorize
from src.sheet_formatter import get_sheet_format
from src.contour_exporter import pixels_to_mm, save_contour


def run_drawing(drawing: Path, output_dir: Path):
    """
    Обработка одного чертежа. 
    """
    image = load_image(drawing)
    print(f"Чертеж детали {drawing.stem} прочитан.")
    print(f"Папка {drawing.parent}")
    image_gray = to_grayscale(image)
    threshold_value, image_binary = binarize(image_gray)

    distance = calculate_distance(image_binary)
    skeleton = calculate_skeleton(image_binary)

    contours = get_contour(distance, skeleton, min_thickness=4)
    num_labels, labels, stats, centroids = (get_components(contours))

    filtered = filter_components(labels, stats, min_area=500, max_rel_size=0.9)

    filtered = remove_roughness_symbol(filtered, labels, stats)

    segments = vectorize(filtered, eps=2.0)

    sheet_format, width_mm, height_mm = get_sheet_format(image, tol=0.02)
    segments_mm = pixels_to_mm(segments, image, width_mm, height_mm)

    print(f"Количество отрезков: {len(segments)}")  

    output_dir.mkdir(parents=True, exist_ok=True)
    
    vector_overlay = image.copy()
    vector_only = np.full_like(image, 255)

    for x1, y1, x2, y2 in segments:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        cv2.line(vector_overlay, p1, p2, (0, 0, 255), 2)
        cv2.line(vector_only, p1, p2, (0, 0, 255), 2)
    
    cv2.imwrite(output_dir / "overlay.png", vector_overlay)
    cv2.imwrite(str(output_dir / "contour_only.png"), vector_only)
    
    save_contour(segments_mm, output_dir / "contour.json")

    print(f"Чертеж детали {drawing.stem} прочитан.")
    print(f"Размер полученного массива: {image.shape}.")
    print(f"Вычисленный трешхолд бинаризации: {threshold_value}.")
    print(f"Пикселей скелета: {np.count_nonzero(skeleton)}")
    print(f"Пикселей после фильтрации: {np.count_nonzero(filtered)}")
    print(f"Формат листа: {sheet_format}")
    print(f"Размер листа: {width_mm}x{height_mm} мм")
    print(f"Количество отрезков: {len(segments_mm)}")
    print("------------------")
