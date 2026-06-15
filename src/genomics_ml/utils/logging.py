"""
Logger setup for the genomics ML pipeline.

Reads logging config (level, file path) from the project's YAML config
and returns a configured logger with console + file handlers.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from genomics_ml.utils.config import load_config, get_config_path


def setup_logger(
    name: str = "genomics_ml",
    level: str = "INFO",
    log_file: Optional[str] = "logs/pipeline.log",
) -> logging.Logger:
    """TODO: Configure and return a logger with console and file handlers.

    Steps to implement:
        1. Get or create a logger with `logging.getLogger(name)`.
        2. Set its level using `getattr(logging, level.upper(), logging.INFO)`.
        3. If the logger already has handlers, return early (avoid duplicates).
        4. Create a Formatter with a timestamp, logger name, level, and message.
        5. Create a StreamHandler(sys.stdout) for console output and attach it.
        6. If log_file is set, create the parent directory, create a FileHandler,
           and attach it.
        7. Return the logger.
    """
    # YOUR CODE HERE
    logger = logging.getLogger(name=name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.hasHandlers():
        return logger
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def get_logger(name: str = "genomics_ml") -> logging.Logger:
    """TODO: Load config, extract logging settings, and call setup_logger.

    Steps:
        1. Use `load_config(get_config_path())` from genomics_ml.utils.config.
        2. Get the "logging" subsection from the config dict.
        3. Call setup_logger with name, level, and log_file from config.
        4. Return the logger.
    """
    configs = load_config(get_config_path())  # YOUR CODE HERE
    logging_config = configs.get("logging")
    logger = setup_logger(
        name=name,
        level=logging_config.get("level"),
        log_file=logging_config.get("file"),
    )
    return logger
