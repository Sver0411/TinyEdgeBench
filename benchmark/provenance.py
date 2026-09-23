"""Git provenance checks shared by the hardware measurement tools."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRTY_ERROR = ("Hardware benchmark requires a clean Git working tree.\n"
               "Commit or stash all changes before producing official results.")


def git_state(root: Path = ROOT) -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                            text=True, capture_output=True, check=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"],
                            cwd=root, text=True, capture_output=True, check=True).stdout
    return commit, bool(status.strip())


def require_clean(allow_dirty: bool = False, root: Path = ROOT) -> tuple[str, bool]:
    commit, dirty = git_state(root)
    if dirty and not allow_dirty:
        raise SystemExit(DIRTY_ERROR)
    return commit, dirty
