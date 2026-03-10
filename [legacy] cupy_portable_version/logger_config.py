import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_file: str = "logs/aste.log", level: int = logging.INFO):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)
    handler_file = logging.FileHandler(log_file)
    handler_file.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.hasHandlers():
        logger.addHandler(handler_console)
        logger.addHandler(handler_file)
    logger.propagate = False
    return logger
