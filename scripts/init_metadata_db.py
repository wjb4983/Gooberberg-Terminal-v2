#!/usr/bin/env python
"""Initialize the platform SQLite metadata catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_platform.data.storage.catalog import (
    DEFAULT_METADATA_DB_PATH,
    init_metadata_db,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_METADATA_DB_PATH,
        help="SQLite metadata database path.",
    )
    return parser.parse_args()


def main() -> None:
    """Create metadata tables."""

    db_path = init_metadata_db(parse_args().path)
    print(f"Initialized metadata catalog at {db_path}")


if __name__ == "__main__":
    main()
