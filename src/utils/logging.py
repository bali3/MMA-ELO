"""Centralized logging setup."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide logging format and threshold.

    Assumption: one root logging configuration is sufficient until services are split.
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
