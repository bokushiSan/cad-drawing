from pathlib import Path


def get_drawings(input_dir: Path) -> list:
    """
    Получение .jpg чертежей во входной директории.

    Args:
        input_dir: Путь к  директории с чертежами.

    Returns:
        Список с путями к чертежам.
    """
    files = [path for path in input_dir.rglob('*') if path.is_file() and path.suffix in [".jpg", ".jpeg"]]
    return files
