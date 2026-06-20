"""Identifier helpers for jobs, tasks, and model artifacts."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str | None = None) -> str:
    """Return a random hex identifier, optionally prefixed with ``<prefix>_``."""

    value = uuid4().hex
    if not prefix:
        return value
    normalized_prefix = prefix.strip().lower().replace("-", "_")
    return f"{normalized_prefix}_{value}"


def new_job_id() -> str:
    """Return a random job identifier."""

    return new_id("job")


def new_task_id() -> str:
    """Return a random task identifier."""

    return new_id("task")


def new_model_id() -> str:
    """Return a random model identifier."""

    return new_id("model")
