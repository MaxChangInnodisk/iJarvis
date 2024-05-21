import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

ROOT = Path("/opt/inno/ijarvis")
LOGGER_NAME = "uvicorn.critical"
LOG_FOLDER = ROOT / "logs"
LOG_FOLDER.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_FOLDER / "ijarvis.log"


def hmesg_wrapper(head: str) -> Callable[[str], str]:
    return lambda text: f"{head.upper()} {text.capitalize()}"


def get_rotation_file_handler(log_path: str) -> RotatingFileHandler:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-.4s] %(message)s (%(filename)s:%(lineno)s)",
        "%y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_path,
        mode="a",
        encoding="utf-8",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        delay=0,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    return file_handler


iLogger: logging.Logger = logging.getLogger(LOGGER_NAME)
iLogger.addHandler(get_rotation_file_handler(LOG_PATH))
