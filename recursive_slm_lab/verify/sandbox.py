from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from .docker import run_in_docker_sandbox


@dataclass
class SandboxResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int


def run_in_sandbox(
    solution_code: str,
    test_code: str,
    assert_tests: list[str] | None = None,
    timeout_s: int = 8,
) -> SandboxResult:
    """Run pytest in a temporary directory with best-effort isolation.

    Note: Python sandboxing is not perfectly safe. This applies a best-effort
    isolation via temp dirs, environment scrubbing, and no-network hints.
    """
    config = Config()
    if config.verify_mode == "docker":
        result = run_in_docker_sandbox(
            solution_code,
            test_code,
            assert_tests=assert_tests,
            timeout_s=timeout_s,
        )
        return SandboxResult(
            passed=result.passed,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )
    strict_mode = os.environ.get("RSLM_STRICT_VERIFY") == "1"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        (temp_path / "solution.py").write_text(solution_code, encoding="utf-8")
        use_fast = os.environ.get("RSLM_FAST_VERIFY") == "1" and assert_tests
        if use_fast:
            harness_lines = ["from solution import *", "", "def run():"] + [
                f"    {assert_test}" for assert_test in assert_tests
            ]
            harness_lines.append("")
            harness_lines.append("if __name__ == '__main__':")
            harness_lines.append("    run()")
            (temp_path / "harness.py").write_text("\n".join(harness_lines), encoding="utf-8")
        else:
            (temp_path / "test_solution.py").write_text(test_code, encoding="utf-8")

        env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
            "PYTHONHASHSEED": "0",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(temp_path),
        }

        command = (
            [sys.executable, "harness.py"]
            if use_fast
            else [sys.executable, "-m", "pytest", "-q", "-p", "no:cov", "-p", "no:ddtrace"]
        )

        preexec_fn = None
        if os.name == "posix":
            import resource

            def _apply_limits() -> None:
                cpu_limit = timeout_s + 1
                memory_limit = 512 * 1024 * 1024 if strict_mode else 1024 * 1024 * 1024
                file_limit = 2 * 1024 * 1024 if strict_mode else 4 * 1024 * 1024
                proc_limit = 8 if strict_mode else 16
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
                resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
                resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
                resource.setrlimit(resource.RLIMIT_NPROC, (proc_limit, proc_limit))

            preexec_fn = _apply_limits

        proc = subprocess.Popen(
            command,
            cwd=temp_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            stdout, stderr = proc.communicate()
            return SandboxResult(
                passed=False,
                stdout=stdout or "",
                stderr=f"Timeout after {timeout_s}s (process group killed)",
                returncode=124,
            )
        return SandboxResult(
            passed=proc.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            returncode=proc.returncode,
        )
