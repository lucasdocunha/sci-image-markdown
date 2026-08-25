"""
Logging setup using Python logging and Rich.
"""

import logging
import sys
from rich.logging import RichHandler


def setup_logger(name: str = "sci-image-markdown", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a rich logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger
