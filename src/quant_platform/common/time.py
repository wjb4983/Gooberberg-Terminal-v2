"""Timezone-aware time helpers."""

from __future__ import annotations

from datetime import date, datetime, timezone

UTC = timezone.utc


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(UTC)


def to_utc(value: datetime) -> datetime:
    """Convert a datetime to timezone-aware UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def iso_utc(value: datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp with a ``Z`` suffix."""

    timestamp = to_utc(value or utc_now())
    return timestamp.isoformat().replace("+00:00", "Z")


def parse_date(value: str | date | datetime) -> date:
    """Parse common date-like inputs into a ``date``."""

    if isinstance(value, datetime):
        return to_utc(value).date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


__all__ = ["UTC", "timezone", "utc_now", "to_utc", "iso_utc", "parse_date"]
