import numpy as np
import cv2
from pathlib import Path
from src.image_loader import load_image
from src.preprocessor import to_grayscale, binarize
from src.line_analyzer import calculate_distance, calculate_skeleton
from src.contour_extractor import get_contour
from src.component_analyzer import get_components, filter_components


def run_drawing(drawing: Path, output_dir: Path):
    """
    Обработка одного чертежа. 
    """
    image = load_image(drawing)
    image_gray = to_grayscale(image)
    threshold_value, image_binary = binarize(image_gray)

    distance = calculate_distance(image_binary)
    skeleton = calculate_skeleton(image_binary)

    contours = get_contour(distance, skeleton, min_thickness=4)
    num_labels, labels, stats, centroids = (get_components(contours))

    filtered = filter_components(labels, stats, min_area=500, max_rel_size=0.9)

    overlay = image.copy()
    overlay[filtered > 0] = (0, 0, 255)

    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_dir / "overlay.png", overlay)

    print(f"Чертеж детали {drawing.stem} прочитан.")
    print(f"Размер полученного массива: {image.shape}.")
    print(f"Вычисленный трешхолд бинаризации: {threshold_value}.")
    print(f"Пикселей скелета: {np.count_nonzero(skeleton)}")
    print(f"Пикселей после фильтрации: {np.count_nonzero(filtered)}")
    print("------------------")


if __name__ == "__main__":
    run_drawing()