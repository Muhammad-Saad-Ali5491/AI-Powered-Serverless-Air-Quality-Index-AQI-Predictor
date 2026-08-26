"""Consistent logging setup used across all pipeline scripts."""
import logging
import sys
from src.utils.paths import LOGS_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        file_handler = logging.FileHandler(LOGS_DIR / "pipeline.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass  # e.g. read-only filesystem in some CI runners; stream logging still works

    return logger
