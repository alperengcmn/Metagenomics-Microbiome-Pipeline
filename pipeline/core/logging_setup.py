"""Logging configuration for the metagenomics pipeline."""
from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(output_dir: Path, verbose: bool = False) -> logging.Logger:
    """Configure pipeline logging with Rich console output and a file handler.

    Args:
        output_dir: Directory where ``pipeline.log`` will be written.
        verbose: If True, set log level to DEBUG; otherwise INFO.

    Returns:
        Configured root logger for the pipeline namespace.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.DEBUG if verbose else logging.INFO

    logger = logging.getLogger("pipeline")
    logger.setLevel(log_level)

    # Avoid duplicate handlers when called more than once
    if logger.handlers:
        logger.handlers.clear()

    # Console handler — use Rich if available, fallback to plain StreamHandler
    try:
        from rich.logging import RichHandler

        console_handler = RichHandler(
            level=log_level,
            rich_tracebacks=True,
            show_time=True,
            show_path=verbose,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except ImportError:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    logger.addHandler(console_handler)

    # File handler — always plain text for easy grepping
    log_path = output_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    logger.debug("Logging initialised. Log file: %s", log_path)
    return logger
