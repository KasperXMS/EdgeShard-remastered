"""Structured logging setup for EdgeShard."""

from __future__ import annotations

import logging
import sys

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    component: str = "edgeshard",
) -> None:
    """Configure structured logging for an EdgeShard process.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON lines (for production / log aggregation).
        component: Component name included in log records (e.g. "master", "worker").
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    if json_output:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f'{{"ts":"%(asctime)s","level":"%(levelname)s","component":"{component}",'
            '"logger":"%(name)s","message":"%(message)s"}}'
        )
        handler.setFormatter(formatter)
    else:
        handler = RichHandler(
            level=log_level,
            show_time=True,
            show_path=False,
            markup=True,
        )

    root = logging.getLogger("edgeshard")
    root.setLevel(log_level)
    root.addHandler(handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the edgeshard namespace."""
    if not name.startswith("edgeshard."):
        name = f"edgeshard.{name}"
    return logging.getLogger(name)
