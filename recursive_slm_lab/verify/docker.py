from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_TAG = "rslm-verify:latest"
DOCKERFILE_PATH = Path(__file__).resolve().parents[2] / "Dockerfile.verify"


@dataclass
class DockerSandboxResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int


def ensure_docker_available() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required for verify_mode=docker but was not found.")


def ensure_docker_image() -> None:
    ensure_docker_available()
    inspect = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
    )
    if inspect.returncode == 0:
        return
    subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, "-f", str(DOCKERFILE_PATH), "."],
        check=True,
        text=True,
    )


def build_docker_command(
    workdir: Path,
    use_fast: bool,
    timeout_s: int,
    strict_mode: bool,
) -> list[str]:
    memory_limit = "512m" if strict_mode else "1g"
    cpu_limit = "1.0" if strict_mode else "2.0"
    pids_limit = "128" if strict_mode else "256"
    tmpfs_size = "64m" if strict_mode else "128m"
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--memory",
        memory_limit,
        "--cpus",
        cpu_limit,
        "--pids-limit",
        pids_limit,
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,size={tmpfs_size}",
        "--env",
        "PYTHONHASHSEED=0",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        "--env",
        "HOME=/tmp",
        "--workdir",
        "/work",
        "-v",
        f"{workdir}:/work:ro",
        IMAGE_TAG,
    ]
    if use_fast:
        command.extend(["python", "-B", "harness.py"])
    else:
        command.extend(
            [
                "python",
                "-B",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:cov",
                "-p",
                "no:ddtrace",
            ]
        )
    return command


def run_in_docker_sandbox(
    solution_code: str,
    test_code: str,
    assert_tests: list[str] | None = None,
    timeout_s: int = 8,
) -> DockerSandboxResult:
    ensure_docker_image()
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

        command = build_docker_command(temp_path, use_fast, timeout_s, strict_mode)
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_s + 2,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return DockerSandboxResult(
                passed=False,
                stdout="",
                stderr=f"Timeout after {timeout_s}s (docker run exceeded limit)",
                returncode=124,
            )
        return DockerSandboxResult(
            passed=proc.returncode == 0,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            returncode=proc.returncode,
        )
