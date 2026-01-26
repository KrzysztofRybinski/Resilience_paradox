"""Logging setup for pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    handlers = [RichHandler(rich_tracebacks=True)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
    return logging.getLogger("resilience_paradox")
