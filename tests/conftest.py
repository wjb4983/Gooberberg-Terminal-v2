"""Project-level pytest configuration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    """Accept pytest-timeout's option when the plugin is unavailable locally."""

    try:
        parser.addoption(
            "--timeout",
            action="store",
            default=None,
            help="Compatibility timeout option for local test runs.",
        )
    except ValueError:
        # pytest-timeout is installed and already registered the option.
        pass
