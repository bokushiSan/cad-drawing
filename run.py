from pathlib import Path
import sys
from src.pipeline import run_drawing
from src.drawing_finder import get_drawings


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
        result_dir = output_dir / drawing.relative_to(input_dir)
        run_drawing(drawing, result_dir)


if __name__ == "__main__":
    main()