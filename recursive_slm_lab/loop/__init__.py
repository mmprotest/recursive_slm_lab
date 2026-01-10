from __future__ import annotations

from typing import TYPE_CHECKING

# Safe re-exports (do not import self_improve here)
from .iteration import run_iteration, IterationResult  # noqa: F401
from .sampling import generate_candidates, Candidate  # noqa: F401
from .prompts import build_prompt  # noqa: F401

if TYPE_CHECKING:
    from .self_improve import run_self_improve as run_self_improve  # pragma: no cover

__all__ = [
    "run_iteration",
    "IterationResult",
    "run_self_improve",
    "generate_candidates",
    "Candidate",
    "build_prompt",
]


def __getattr__(name: str):
    if name == "run_self_improve":
        from .self_improve import run_self_improve

        return run_self_improve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
