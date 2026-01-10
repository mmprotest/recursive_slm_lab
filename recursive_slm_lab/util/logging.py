from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str | int = logging.INFO) -> logging.Logger:
    if isinstance(level, str):
        resolved = logging._nameToLevel.get(level.upper())
        if resolved is None:
            raise ValueError(f"Unsupported log level: {level}")
        level = resolved
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    return logging.getLogger("rslm")
