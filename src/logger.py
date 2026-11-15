import logging
from pathlib import Path


def create_logger(output_dir: Path) -> logging.Logger:
    logger = logging.Logger("drawing")

    handler = logging.FileHandler(output_dir, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)

    return logger


def close_logger(logger: logging.Logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
