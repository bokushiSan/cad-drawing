from src.drawing_finder import get_drawings
from src.image_loader import load_image
from src.preprocessor import to_grayscale, binarize
from pathlib import Path
import sys
import cv2


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
        print(f"Чертеж детали {drawing.stem} прочитан.")
        print(f"Размер полученного массива: {image.shape}.")
        print(f"Размер бинарного массива: {image_binary.shape}. Трешхолд: {threshold_value}.")
        
        drawing_parent_dir = drawing.relative_to(input_dir).parent
        drawing_dir = output_dir / drawing_parent_dir
        drawing_dir.mkdir(parents=True, exist_ok=True)
        result_path = drawing_dir / f"{drawing.stem}.png"

        cv2.imwrite(result_path, image_binary)

        print(f"Чертеж детали {drawing.stem} сохранен.")
        print("---------------")

if __name__ == "__main__":
    main()