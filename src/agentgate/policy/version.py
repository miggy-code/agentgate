"""Policy versions: a content hash of the bundle and the git commit it came from.

The content hash is what every decision records, so an audit entry can be tied to the exact
rules that produced it. It is computed from the validated bundle rather than the YAML text,
so comments, whitespace and key order do not change it, while any change in meaning does.
"""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from decimal import Decimal
from enum import Enum
from pathlib import Path

from agentgate.policy.schema import PolicyBundle

_GIT_TIMEOUT_SECONDS = 5


def bundle_version(bundle: PolicyBundle) -> str:
    """Return "sha256:" plus the first 16 hex chars of the canonical bundle hash."""
    # Python mode keeps sets distinguishable from ordered tuples, so sets can be sorted
    # while rule and label order, which is meaningful, is kept as written.
    canonical = _canonical(bundle.model_dump(mode="python", by_alias=True))
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("ascii")).hexdigest()[:16]


type _Json = bool | int | str | list[_Json] | dict[str, _Json] | None


def _canonical(value: object) -> _Json:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return _decimal(value)
    if isinstance(value, set | frozenset):
        items = [_canonical(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, list | tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    raise TypeError(f"cannot canonicalise {type(value).__name__}")


def _decimal(value: Decimal) -> str:
    # "0.50", "0.5" and "5E-1" are the same limit, so they must hash the same.
    if value.is_zero():
        return "0"
    return format(value.normalize(), "f")


def git_revision(path: Path) -> str | None:
    """Short hash of the last commit touching path, or None if that cannot be stated honestly.

    None covers: not a git repo, git missing or failing, the path never committed, and the
    path having uncommitted changes (the commit would then not describe what is loaded).
    """
    try:
        target = path.resolve()
        cwd = target if target.is_dir() else target.parent
        status = _git(cwd, "status", "--porcelain", "--", str(target))
        if status is None or status.strip():
            return None
        revision = _git(cwd, "log", "-1", "--format=%h", "--", str(target))
        if revision is None:
            return None
        return revision.strip() or None
    except (OSError, ValueError, RuntimeError):
        return None


def _git(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, UnicodeDecodeError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout
