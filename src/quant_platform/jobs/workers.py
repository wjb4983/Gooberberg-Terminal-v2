"""Local RQ worker entry points for platform jobs."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rq import Worker

from quant_platform.config.settings import get_settings
from quant_platform.jobs.queue import DEFAULT_QUEUE_NAME, job_queue, redis_connection


def build_worker(queue_names: Sequence[str] | None = None) -> Worker:
    """Build an RQ worker for the configured Redis connection."""

    settings = get_settings()
    connection = redis_connection(settings)
    names = tuple(queue_names or (DEFAULT_QUEUE_NAME,))
    queues = [job_queue(name, connection=connection) for name in names]
    return Worker(queues, connection=connection)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the development worker."""

    parser = argparse.ArgumentParser(description="Run a quant platform RQ worker.")
    parser.add_argument(
        "queues",
        nargs="*",
        default=[DEFAULT_QUEUE_NAME],
        help="Queue names to consume, in priority order.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run a worker until interrupted."""

    args = parse_args(argv)
    build_worker(args.queues).work()


if __name__ == "__main__":
    main()


__all__ = ["build_worker", "main", "parse_args"]
