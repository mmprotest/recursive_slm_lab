from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from recursive_slm_lab.eval.strength import is_task_weak
from recursive_slm_lab.tasks import Task


def test_strength_probe_flags_weak_task() -> None:
    task = Task(
        task_id="weak-1",
        prompt="Return the input number.",
        function_name="identity",
        signature="(n)",
        reference_tests=(
            "import pytest\n"
            "from solution import *\n\n"
            "def test_basic():\n"
            "    assert identity(1) in [0, 1]\n"
        ),
        category="math",
        difficulty=1,
        assert_tests=["assert identity(1) in [0, 1]"],
    )
    assert is_task_weak(task)
