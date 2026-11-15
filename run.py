from pathlib import Path
import sys
from src.pipeline import run_drawing
from src.drawing_finder import get_drawings
from src.logger import create_logger, close_logger
from src.report import save_report


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

    results = []

    for drawing in drawings:
        result_dir = output_dir / drawing.relative_to(input_dir).parent / drawing.stem
        result_dir.mkdir(parents=True, exist_ok=True)

        logger = create_logger(result_dir / "log.txt")

        try:
            verdict, reason = run_drawing(drawing, result_dir, logger)

        except Exception as e:
            verdict = "red"
            reason = f"Ошибка обработки: {e}"
            logger.exception(reason)
            print(f"{reason = }")
            (result_dir / "verdict.txt").write_text(f"{verdict}: {reason}", encoding="utf-8")

        finally:
            close_logger(logger)

        results.append({
            "drawing": str(drawing.relative_to(input_dir)),
            "verdict": verdict,
            "reason": reason,
        })

        print(f"{drawing.name}: {verdict}")

    save_report(results, output_dir)


if __name__ == "__main__":
    main()