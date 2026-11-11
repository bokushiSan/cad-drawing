from pathlib import Path
import cv2
import numpy as np


def load_image(image_path: Path) -> np.ndarray:
    """
    Загрузка изображения чертежа из файла.

    Args:
        image_path: Путь к чертежу.

    Returns:
        Изображение в виде numpy массива.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Не удалось загрузить чертеж детали {image_path.stem}")

    return image


if __name__ == '__main__':
    image_path = Path("/home/bokushi/Загрузки/исходники_растр/229_stage/Патрубок _ В307.622.03.00.01.jpg")
    image = load_image(image_path)
    print(f"Чертеж детали {image_path.stem} загружен. Размер полученного массива: {image.shape}")
