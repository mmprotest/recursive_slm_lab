from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def apply_patch_with_git(repo_root: Path, diff_text: str, workdir: Path) -> None:
    has_git = (repo_root / ".git").exists()
    workdir.mkdir(parents=True, exist_ok=True)
    if has_git:
        result = subprocess.run(
            ["git", "worktree", "add", "--detach", str(workdir)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {result.stderr.strip()}")
    else:
        shutil.copytree(repo_root, workdir, dirs_exist_ok=True)

    patch_path = workdir / "self_patch.diff"
    patch_path.write_text(diff_text, encoding="utf-8")
    result = subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git apply failed: {result.stderr.strip()}")
