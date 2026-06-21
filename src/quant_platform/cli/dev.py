"""Run the local Gooberberg development stack with one command."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ManagedProcess:
    name: str
    command: Sequence[str]
    env: dict[str, str] | None = None


def _run_setup_command(command: Sequence[str]) -> None:
    print(f"[setup] {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def _start_process(process: ManagedProcess) -> subprocess.Popen[bytes]:
    print(f"[start] {process.name}: {' '.join(process.command)}", flush=True)
    return subprocess.Popen(process.command, env=process.env)


def _terminate_processes(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.monotonic() + 10
    for process in processes:
        remaining = max(0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Redis, the FastAPI backend, the Streamlit UI, "
            "and the RQ worker."
        )
    )
    parser.add_argument(
        "--skip-db-init",
        action="store_true",
        help="Do not run scripts/init_metadata_db.py before starting services.",
    )
    parser.add_argument(
        "--skip-redis",
        action="store_true",
        help="Do not start Redis with Docker Compose before starting services.",
    )
    parser.add_argument(
        "--api-port",
        default="8000",
        help="Port for the FastAPI service. Defaults to 8000.",
    )
    parser.add_argument(
        "--ui-port",
        default="8501",
        help="Port for the Streamlit UI. Defaults to 8501.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.skip_db_init:
        _run_setup_command([sys.executable, "scripts/init_metadata_db.py"])

    if not args.skip_redis:
        _run_setup_command(["docker", "compose", "up", "-d", "redis"])

    env = os.environ.copy()
    env.setdefault("QUANT_PLATFORM_REDIS_URL", "redis://localhost:6379/0")
    env.setdefault("QUANT_PLATFORM_API_BASE_URL", f"http://localhost:{args.api_port}")

    services = [
        ManagedProcess(
            name="api",
            command=[
                sys.executable,
                "-m",
                "uvicorn",
                "apps.api.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                args.api_port,
                "--reload",
            ],
            env=env,
        ),
        ManagedProcess(
            name="ui",
            command=[
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "apps/ui/Home.py",
                "--server.address",
                "0.0.0.0",
                "--server.port",
                args.ui_port,
            ],
            env=env,
        ),
        ManagedProcess(
            name="worker",
            command=[sys.executable, "-m", "quant_platform.jobs.workers"],
            env=env,
        ),
    ]

    processes: list[subprocess.Popen[bytes]] = []

    def stop_stack(signum: int, _frame: object) -> None:
        print(f"[stop] received signal {signum}; stopping services", flush=True)
        _terminate_processes(processes)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, stop_stack)
    signal.signal(signal.SIGTERM, stop_stack)

    try:
        processes = [_start_process(service) for service in services]
        ready_message = (
            f"[ready] API: http://localhost:{args.api_port}/docs | "
            f"UI: http://localhost:{args.ui_port}"
        )
        print(ready_message, flush=True)
        while True:
            for process, service in zip(processes, services, strict=True):
                return_code = process.poll()
                if return_code is not None:
                    stop_message = (
                        f"[stop] {service.name} exited with code {return_code}; "
                        "stopping stack"
                    )
                    print(stop_message, flush=True)
                    _terminate_processes(processes)
                    return return_code
            time.sleep(1)
    finally:
        _terminate_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
