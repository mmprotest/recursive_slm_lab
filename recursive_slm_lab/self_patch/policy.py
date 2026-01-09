from __future__ import annotations

FORBIDDEN_PREFIXES = (
    "recursive_slm_lab/eval/",
    "recursive_slm_lab/verify/",
    "recursive_slm_lab/training/promotion.py",
    "recursive_slm_lab/meta/policy_improve.py",
    "recursive_slm_lab/loop/self_improve.py",
)


def validate_patch_touches(diff_text: str) -> None:
    touched = _extract_paths(diff_text)
    forbidden = [path for path in touched if _is_forbidden(path)]
    if forbidden:
        formatted = ", ".join(sorted(forbidden))
        raise ValueError(f"Patch touches forbidden paths: {formatted}")


def _extract_paths(diff_text: str) -> set[str]:
    paths: set[str] = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                paths.update({_normalize_path(parts[2]), _normalize_path(parts[3])})
        elif line.startswith("--- "):
            paths.add(_normalize_path(line[4:].strip()))
        elif line.startswith("+++ "):
            paths.add(_normalize_path(line[4:].strip()))
    return {path for path in paths if path and path != "/dev/null"}


def _normalize_path(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]
    return path


def _is_forbidden(path: str) -> bool:
    for prefix in FORBIDDEN_PREFIXES:
        if prefix.endswith("/") and path.startswith(prefix):
            return True
        if path == prefix:
            return True
    return False
