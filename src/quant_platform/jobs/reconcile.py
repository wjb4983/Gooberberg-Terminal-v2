"""Maintenance CLI for reconciling catalog jobs with RQ state."""

from __future__ import annotations

import argparse
import json

from quant_platform.common.enums import JobStatus
from quant_platform.jobs.queue import reconcile_stale_jobs


def main() -> None:
    """Run catalog/RQ reconciliation and print a JSON summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        dest="statuses",
        action="append",
        help="Catalog status to inspect; repeat for multiple statuses.",
    )
    parser.add_argument(
        "--stale-status",
        choices=[JobStatus.CANCELLED.value, JobStatus.FAILED.value],
        default=JobStatus.CANCELLED.value,
        help="Terminal status to apply when an RQ job id is stale.",
    )
    args = parser.parse_args()
    result = reconcile_stale_jobs(
        statuses=set(args.statuses) if args.statuses else None,
        stale_status=JobStatus(args.stale_status),
    )
    print(
        json.dumps(
            {
                "checked": result.checked,
                "reconciled": [item.__dict__ for item in result.reconciled],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
