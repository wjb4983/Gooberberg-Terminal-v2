"""Path helpers for platform storage locations."""

from __future__ import annotations

from pathlib import Path


def expand_path(path: str | Path) -> Path:
    """Return an absolute path with user home and environment variables expanded."""

    return Path(path).expanduser().resolve()


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return its absolute path."""

    resolved = expand_path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the file path."""

    resolved = expand_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def data_lake_subpath(root: str | Path, *parts: str) -> Path:
    """Build a path inside the configured data lake root."""

    return expand_path(root).joinpath(*parts)
