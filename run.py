from src.drawing_finder import get_drawings
from src.image_loader import load_image
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 3:
        print("Запуск должен быть: run.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_dir.exists():
        print("Папка input не найдена")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    drawings = get_drawings(input_dir)

    print(f"Найдено {len(drawings)} чертежей в папке {input_dir}")

    for drawing in drawings:
        image = load_image(drawing)
        print(f"Чертеж детали {drawing.stem} загружен. Размер полученного массива: {image.shape}")

if __name__ == "__main__":
    main()