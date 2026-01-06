from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int


def run_in_sandbox(solution_code: str, test_code: str, timeout_s: int = 8) -> SandboxResult:
    """Run pytest in a temporary directory with best-effort isolation.

    Note: Python sandboxing is not perfectly safe. This applies a best-effort
    isolation via temp dirs, environment scrubbing, and no-network hints.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "solution.py").write_text(solution_code, encoding="utf-8")
        (temp_path / "test_solution.py").write_text(test_code, encoding="utf-8")

        env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "NO_NETWORK": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(temp_path),
        }

        try:
            proc = subprocess.run(
                ["python", "-m", "pytest", "-q", "-p", "no:cov", "-p", "no:ddtrace"],
                cwd=temp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return SandboxResult(
                passed=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
            )
        except subprocess.TimeoutExpired as exc:
            return SandboxResult(
                passed=False,
                stdout=exc.stdout or "",
                stderr=f"Timeout after {timeout_s}s",
                returncode=124,
            )
