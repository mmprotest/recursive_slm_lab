from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from ..llm.base import LLMBackend
from ..util import stable_timestamp, ensure_dir
from .apply import apply_patch_with_git
from .gates import run_patch_gates
from .models import PatchGateResult, PatchPromotionResult
from .policy import validate_patch_touches
from .propose import propose_patch


def run_self_patch(
    backend: LLMBackend,
    repo_root: Path,
    db_path: str,
    context: dict,
    artifacts_dir: Path,
    heldout_size: int,
    heldout_limit: int | None,
    regression_size: int,
    seed: int,
    apply_patch: bool,
) -> PatchPromotionResult:
    db_path = str(Path(db_path).resolve())
    proposal = None
    gate_result = PatchGateResult(passed=False, steps=[])
    applied = False
    committed = False
    message = "Self-patch skipped."
    error_message = None

    try:
        proposal = propose_patch(backend, repo_root, context)
        validate_patch_touches(proposal.diff_text)
        workdir_path = Path(tempfile.mkdtemp(prefix="rslm_self_patch_"))
        try:
            apply_patch_with_git(repo_root, proposal.diff_text, workdir_path)
            gate_result = run_patch_gates(
                workdir_path,
                db_path=db_path,
                heldout_size=heldout_size,
                heldout_limit=heldout_limit,
                regression_size=regression_size,
                seed=seed,
            )
        finally:
            _cleanup_workdir(repo_root, workdir_path)
        if gate_result.passed and apply_patch:
            applied = _apply_patch_to_repo(repo_root, proposal.diff_text)
            if applied:
                committed = _commit_patch(repo_root)
                message = "Self-patch promoted." if committed else "Self-patch applied without commit."
            else:
                message = "Self-patch gates passed but patch was not applied."
        elif gate_result.passed:
            message = "Self-patch gates passed (dry run)."
        else:
            message = "Self-patch rejected by gates."
    except Exception as exc:
        error_message = str(exc)
        gate_result = PatchGateResult(passed=False, steps=[], error=error_message)
        message = f"Self-patch failed: {error_message}"

    artifact_path = _write_artifact(
        artifacts_dir,
        proposal=proposal,
        gate_result=gate_result,
        applied=applied,
        committed=committed,
        message=message,
    )
    return PatchPromotionResult(
        proposal=proposal,
        gate_result=gate_result,
        applied=applied,
        committed=committed,
        message=message,
        artifact_path=artifact_path,
    )


def _cleanup_workdir(repo_root: Path, workdir: Path) -> None:
    if not workdir.exists():
        return
    if (repo_root / ".git").exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(workdir)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    else:
        shutil.rmtree(workdir, ignore_errors=True)


def _apply_patch_to_repo(repo_root: Path, diff_text: str) -> bool:
    if not (repo_root / ".git").exists():
        return False
    patch_path = repo_root / "self_patch.diff"
    patch_path.write_text(diff_text, encoding="utf-8")
    try:
        result = subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git apply failed: {result.stderr.strip()}")
        return True
    finally:
        patch_path.unlink(missing_ok=True)


def _commit_patch(repo_root: Path) -> bool:
    if not (repo_root / ".git").exists():
        return False
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        return False
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=False)
    result = subprocess.run(
        ["git", "commit", "-m", "self-patch: automated update"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _write_artifact(
    artifacts_dir: Path,
    proposal,
    gate_result: PatchGateResult,
    applied: bool,
    committed: bool,
    message: str,
) -> Path:
    ensure_dir(artifacts_dir)
    timestamp = stable_timestamp()
    path = artifacts_dir / f"patch_{timestamp}.json"
    payload = {
        "proposal": asdict(proposal) if proposal else None,
        "gate_result": asdict(gate_result),
        "applied": applied,
        "committed": committed,
        "message": message,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
