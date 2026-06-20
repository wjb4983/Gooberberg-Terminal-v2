# Gooberberg Terminal v2

Python quant platform scaffold with a FastAPI backend, Streamlit UI, reusable `quant_platform` package, local data-lake utilities, Massive.com market-data integration, and Redis/RQ worker plumbing.

## Repository structure

```text
apps/
  api/                 FastAPI application entrypoint
  ui/                  Streamlit application entrypoint
configs/               Versioned non-secret configuration examples
data/                  Local development data placeholder (ignored except .gitkeep)
docs/                  Project documentation
scripts/               Developer and operations scripts
src/quant_platform/    Importable platform package
  backtesting/         Strategy simulation and performance analysis
  common/              Shared enums, IDs, paths, and time helpers
  config/              Pydantic settings and schemas
  data/                Data lake, storage, datasets, and provider utilities
  execution/           Broker/execution integrations
  features/            Feature engineering pipelines
  ingest/              Market/reference data ingestion
  jobs/                RQ task definitions and orchestration
  models/              Model training, inference, and artifacts
  research/            Research workflow helpers
  services/            Application service layer
tests/                 Pytest test suite
```

## Access assumptions

This project assumes the preferred remote-development path is VSCode over SSH to a host that is reachable on your private Tailscale network. Tailscale should provide device-to-device connectivity; the application does not configure or expose a public tunnel for you. Keep FastAPI, Streamlit, Redis, and any worker ports bound to localhost or a trusted Tailscale interface unless you intentionally harden them for broader access.

## Quick start

Run these tasks in order:

1. Connect to the development machine with VSCode Remote - SSH over its Tailscale hostname or Tailscale IP.
2. Open this repository in VSCode on the remote host.
3. Install `uv` if it is not already available: <https://docs.astral.sh/uv/>.
4. Copy environment defaults: `cp .env.example .env`.
5. Set `MASSIVE_API_KEY` in `.env` if you need live Massive.com data. Leave it blank when using fake/mock providers.
6. Fill only local values in `.env`; never commit `.env`, API keys, tokens, credentials, or other secrets.
7. Install dependencies: `uv sync --dev`.
8. Run tests with a timeout: `uv run pytest --timeout=30`.
9. Start optional Redis dependency if you need queued jobs: `docker compose up redis`.
10. Start the API directly with Python tooling: `uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload`.
11. Start the Streamlit UI directly: `uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501`.
12. Start an RQ worker when processing queued jobs: `uv run python -m quant_platform.jobs.workers`.
13. Open the FastAPI docs at <http://localhost:8000/docs> or via your SSH/Tailscale port-forwarding workflow.
14. Open the Streamlit UI at <http://localhost:8501> or via your SSH/Tailscale port-forwarding workflow.

## VSCode SSH workflow

1. Install the VSCode **Remote - SSH** extension locally.
2. Confirm the remote machine is visible in Tailscale and that SSH is allowed on the remote host.
3. Add an SSH target using the Tailscale DNS name or 100.x Tailscale IP, for example `ssh user@dev-host.tailnet-name.ts.net`.
4. Use **Remote-SSH: Connect to Host...**, open `/workspace/Gooberberg-Terminal-v2`, and run terminal commands in the VSCode remote terminal.
5. Forward ports `8000` for FastAPI and `8501` for Streamlit from the VSCode **Ports** panel if you want browser access from your local workstation.

## Environment setup

`.env.example` documents required local variables and intentionally includes only blank or non-secret development defaults. Create your local file with:

```bash
cp .env.example .env
```

Set the Massive.com key only in your local `.env` or shell environment:

```bash
MASSIVE_API_KEY=your_live_key_here
QUANT_PLATFORM_REDIS_URL=redis://localhost:6379/0
QUANT_PLATFORM_DATA_LAKE_ROOT=./data/lake
QUANT_PLATFORM_CATALOG_DB_PATH=./data/catalog/catalog.db
QUANT_PLATFORM_API_BASE_URL=http://localhost:8000
```

Do not commit secrets. Before committing, check `git status --short` and verify `.env` is not staged.

## Direct Python execution commands

Use direct Python/uv commands for day-to-day development:

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501
uv run python -m quant_platform.jobs.workers
```

You can also use the worker helper script:

```bash
./scripts/worker_dev.sh
```

## Optional Docker Compose services

Docker Compose is optional and mainly useful for local dependencies or running the bundled API/UI containers:

```bash
docker compose up redis
docker compose up api
docker compose up ui
docker compose up --build
```

The direct Python commands above remain the recommended development workflow because they make logs, breakpoints, and VSCode debugging simpler.

## Tests and checks with timeouts

All test commands should include explicit timeouts so automation cannot hang indefinitely:

```bash
uv run pytest --timeout=30
uv run pytest tests/unit --timeout=30
uv run ruff check .
uv run mypy src
```

## More documentation

- [Architecture](docs/architecture.md)
- [Data lake](docs/data_lake.md)
- [HPC setup](docs/hpc_setup.md)
- [Massive provider](docs/massive_provider.md)
