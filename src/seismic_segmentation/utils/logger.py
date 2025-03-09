# src/seismic_segmentation/utils/logger.py
"""Logging utilities for seismic segmentation."""

import logging
from pathlib import Path


def get_logger(log_dir, name="seismic_segmentation"):
    """
    Set up logging.

    Args:
        log_dir: Directory for log files
        name: Logger name

    Returns:
        Logger instance
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Only set up handlers if they haven't been set up already
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Create file handler
        file_handler = logging.FileHandler(str(log_dir / "log.txt"))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
