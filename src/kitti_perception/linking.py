from __future__ import annotations

import os
import shutil
from pathlib import Path


def _equivalent(left: Path, right: Path) -> bool:
    try:
        if left.samefile(right):
            return True
    except (FileNotFoundError, OSError):
        pass
    if not left.is_file() or not right.is_file():
        return False
    left_stat = left.stat()
    right_stat = right.stat()
    return left_stat.st_size == right_stat.st_size and left_stat.st_mtime_ns == right_stat.st_mtime_ns


def reference_file(destination: Path, source: Path) -> str:
    source = source.expanduser().resolve()
    destination = destination.expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if not _equivalent(destination, source):
            raise RuntimeError(f"Existing destination differs from source: {destination}")
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        destination.symlink_to(source)
        return "symlink"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"

