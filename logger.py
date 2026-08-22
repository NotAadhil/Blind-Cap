"""
logger.py - Blind Cap Logging Module
=====================================
Configures Python's standard ``logging`` library with:
  - Console handler  → INFO and above (coloured severity prefix)
  - Rotating file    → DEBUG and above  (``logs/blind_cap.log``, max 2 MB, 3 backups)

Usage in any module::

    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("Camera opened")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "blind_cap.log"
MAX_LOG_BYTES = 2 * 1024 * 1024   # 2 MB per log file
BACKUP_COUNT = 3                   # keep 3 rotated backups

_CONFIGURED = False                # ensures setup runs only once


def _setup_root_logger() -> None:
    """Configure the root ``blind_cap`` logger (called once)."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger("blind_cap")
    root.setLevel(logging.DEBUG)

    # Prevent duplicate handlers if module is reloaded
    if root.handlers:
        return

    # --- Console handler (INFO+) ---
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console_fmt = logging.Formatter(
        "[%(levelname)-7s] %(message)s"
    )
    console.setFormatter(console_fmt)
    root.addHandler(console)

    # --- Rotating file handler (DEBUG+) ---
    try:
        file_handler = RotatingFileHandler(
            str(LOG_FILE),
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        root.addHandler(file_handler)
    except OSError as exc:
        # If we can't write to the log file, continue with console only
        root.warning("Could not open log file %s: %s", LOG_FILE, exc)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the ``blind_cap`` namespace.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        A ``logging.Logger`` instance ready to use.

    Example::

        logger = get_logger(__name__)
        logger.info("Model loaded on %s", device)
    """
    _setup_root_logger()
    # Prefix with "blind_cap." so all project loggers share configuration
    if name.startswith("blind_cap."):
        return logging.getLogger(name)
    return logging.getLogger(f"blind_cap.{name}")
