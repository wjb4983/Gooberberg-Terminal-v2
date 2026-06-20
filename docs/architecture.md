# Architecture

Gooberberg Terminal v2 is organized as a modular Python quant platform. The repository separates application entrypoints from the importable package so API, UI, worker, research, and test processes can share the same domain code.

## Runtime components

- **FastAPI API**: `apps/api/main.py` exposes HTTP routes and OpenAPI docs.
- **Streamlit UI**: `apps/ui/Home.py` provides an interactive browser UI for local research and operations.
- **Python package**: `src/quant_platform` contains configuration, data, jobs, backtesting, training, and service logic.
- **Redis/RQ worker**: `quant_platform.jobs.workers` consumes queued jobs from Redis using queue helpers in `quant_platform.jobs.queue`.
- **Metadata catalog**: SQLite-backed catalog state lives under the configured catalog path.
- **Data lake**: local Parquet and manifest artifacts live under the configured data-lake root.

## Access and remote-development assumptions

The recommended workflow is VSCode Remote - SSH into a development host that is reachable over Tailscale. Tailscale is assumed to provide private network reachability; this repository does not configure Tailscale ACLs, SSH policy, public DNS, TLS certificates, or public ingress.

Keep development services private by default. Bind ports only to localhost or a trusted Tailscale interface unless you have reviewed firewall rules, Tailscale ACLs, authentication, and secret handling.

## Recommended quick-start order

1. Connect to the remote development host from VSCode using Remote - SSH over Tailscale.
2. Open the repository in the VSCode remote window.
3. Install or verify `uv`.
4. Copy `.env.example` to `.env`.
5. Add `MASSIVE_API_KEY` to `.env` only if live Massive.com data is required.
6. Confirm no secrets are committed or staged.
7. Install dependencies with `uv sync --dev`.
8. Run `uv run pytest --timeout=60`.
9. Start Redis with `docker compose up redis` if you need RQ jobs.
10. Start the API with `uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload`.
11. Start Streamlit with `uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501`.
12. Start a worker with `uv run python -m quant_platform.jobs.workers` when queued jobs should execute.

## Direct Python execution

Direct Python execution is the preferred development path:

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501
uv run python -m quant_platform.jobs.workers
```

This keeps process ownership in the VSCode terminal, simplifies debugging, and avoids rebuilding containers for every source edit.

## Optional Docker Compose services

Docker Compose can run Redis and the bundled app services when container parity is useful:

```bash
docker compose up redis
docker compose up api
docker compose up ui
docker compose up --build
```

Use Compose selectively. Redis is the most common service to run in Docker while API, UI, and worker processes run directly with `uv`.

## Configuration and secrets

Settings load from environment variables and `.env`. Use:

```bash
cp .env.example .env
```

Then set local values such as:

```bash
MASSIVE_API_KEY=your_live_key_here
QUANT_PLATFORM_REDIS_URL=redis://localhost:6379/0
QUANT_PLATFORM_DATA_LAKE_ROOT=./data/lake
QUANT_PLATFORM_CATALOG_DB_PATH=./data/catalog/catalog.db
QUANT_PLATFORM_API_BASE_URL=http://localhost:8000
```

Never commit `.env` or secrets. Keep `MASSIVE_API_KEY` in your shell, secret manager, or ignored local `.env` file only.

## Tests and checks

Use timeouts for test execution:

```bash
uv run pytest --timeout=60
uv run pytest tests/unit --timeout=60
uv run ruff check .
uv run mypy src
```
