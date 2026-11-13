from src.drawing_finder import get_drawings
from src.image_loader import load_image
from src.preprocessor import to_grayscale, binarize
from src.line_analyzer import calculate_distance, calculate_skeleton, get_thickness, build_overlay_thickness
from src.contour_extractor import get_contour
from pathlib import Path
import sys
import cv2
import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Запуск должен быть: run.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.is_dir():
        print("Папка input не найдена")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    drawings = get_drawings(input_dir)

    print(f"Найдено {len(drawings)} чертежей в папке {input_dir}")

    for drawing in drawings:
        image = load_image(drawing)
        image_gray = to_grayscale(image)
        threshold_value, image_binary = binarize(image_gray)
        print(drawing)
        print(f"Чертеж детали {drawing.stem} прочитан.")
        print(f"Размер полученного массива: {image.shape}.")
        print(f"Размер бинарного массива: {image_binary.shape}. Трешхолд: {threshold_value}.")
        
        drawing_parent_dir = drawing.relative_to(input_dir).parent
        drawing_dir = output_dir / drawing_parent_dir
        drawing_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(drawing_dir / "binary.png", image_binary)

        distance = calculate_distance(image_binary)
        skeleton = calculate_skeleton(image_binary)
        thickness = get_thickness(distance, skeleton)

        countours = get_contour(
            distance,
            skeleton,
            min_thickness=4,
        )
        cv2.imwrite(drawing_dir / "contour.png", cv2.bitwise_not(countours))

        overlay = image.copy()
        overlay[countours > 0] = (0, 0, 255)
        cv2.imwrite(drawing_dir / "contour_overlay.png", overlay)

        skeleton_count = np.count_nonzero(skeleton)
        candidates_count = np.count_nonzero(countours)

        print(f"Пикселей скелета: {skeleton_count}")
        print(f"Пикселей кандидатов: {candidates_count}")

        if skeleton_count > 0:
            ratio = candidates_count / skeleton_count * 100
            print(f"Сохранено пикселей скелета: {ratio:.2f}%")

        if thickness.size > 0:
            print("Линии обнаружены.")
            print(f"Медиана: {np.median(thickness):.2f} px")
            print(f"25-й процентиль: {np.percentile(thickness, 25):.2f} px")
            print(f"75-й процентиль: {np.percentile(thickness, 75):.2f} px")
            print(f"90-й процентиль: {np.percentile(thickness, 90):.2f} px")
            counts, edges = np.histogram(
                thickness,
                bins=[0, 2, 4, 6, 8, 10, 15, 20, np.inf],
            )

            for left, right, count in zip(edges[:-1], edges[1:], counts):
                print(f"{left:.0f}–{right:.0f} px: {count}")

            thickness_overlay = build_overlay_thickness(image, distance, skeleton)

            cv2.imwrite(drawing_dir / "thickness.png", thickness_overlay)

            print("Визуализация толщины сохранена.")
        else:
            print("Линии не обнаружены.")

        thin_count = np.sum(thickness < 4)
        medium_count = np.sum((thickness >= 4) & (thickness < 8))
        thick_count = np.sum(thickness >= 8)

        print(f"Тонкие: {thin_count}")
        print(f"Средние: {medium_count}")
        print(f"Толстые: {thick_count}")

        distance_png = cv2.normalize(distance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cv2.imwrite(drawing_dir / "distance.png", distance_png)

        print(f"Расстояние посчитано.")

        print(f"Чертеж детали {drawing.stem} сохранен.")
        print("---------------")

if __name__ == "__main__":
    main()