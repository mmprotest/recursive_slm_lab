from __future__ import annotations

from contextlib import contextmanager
import time


@contextmanager
def log_timing(logger, label: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("%s completed in %.2fs", label, elapsed)
