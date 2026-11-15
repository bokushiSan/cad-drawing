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
from src.contour_reviewer import review_contour


def run_drawing(drawing: Path, output_dir: Path):
    """
    Обработка одного чертежа. 

    Args:
        drawing: Путь к чертежу.
        output_dir: Папка для сохранения результатов.
    """
    image = load_image(drawing)
    print(f"Чертеж детали {drawing.stem} прочитан.")
    image_gray = to_grayscale(image)
    threshold_value, image_binary = binarize(image_gray)

    distance = calculate_distance(image_binary)
    skeleton = calculate_skeleton(image_binary)

    contours = get_contour(distance, skeleton, min_thickness=4)
    num_labels, labels, stats, centroids = get_components(contours)

    filtered = filter_components(labels, stats, min_area=500, max_rel_size=0.9)

    filtered = remove_roughness_symbol(filtered, labels, stats)

    segments = vectorize(filtered, eps=2.0)

    sheet_format, width_mm, height_mm = get_sheet_format(image, tol=0.02)
    segments_mm = pixels_to_mm(segments, image, width_mm, height_mm)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    vector_overlay = image.copy()
    vector_only = np.full_like(image, 255)

    for x1, y1, x2, y2 in segments:
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        cv2.line(vector_overlay, p1, p2, (0, 0, 255), 5)
        cv2.line(vector_only, p1, p2, (0, 0, 255), 5)
    
    cv2.imwrite(output_dir / "overlay.png", vector_overlay)
    cv2.imwrite(output_dir / "contour_only.png", vector_only)
    
    save_contour(segments_mm, output_dir / "contour.json")

    verdict, reason = review_contour(drawing, output_dir / "contour_only.png")
    verdict_text = f"{verdict}: {reason}"

    (output_dir / "verdict.txt").write_text(verdict_text, encoding="utf-8")

    print(f"Вердикт: {verdict_text}")
    print("------------------")
