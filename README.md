# Gooberberg Terminal v2

Initial scaffold for a Python quant platform with a FastAPI backend, Streamlit UI, reusable `quant_platform` package, and local Redis-backed worker dependencies.

## Repository structure

```text
apps/
  api/                 FastAPI application entrypoint
  ui/                  Streamlit application entrypoint
configs/               Versioned non-secret configuration examples
data/                 Local development data placeholder (ignored except .gitkeep)
docs/                  Project documentation
scripts/               Developer and operations scripts
src/quant_platform/    Importable platform package
  backtesting/         Strategy simulation and performance analysis
  common/              Shared enums, IDs, paths, and time helpers
  config/              Pydantic settings and schemas
  data/                Data lake and storage utilities
  execution/           Broker/execution integrations
  features/            Feature engineering pipelines
  ingest/              Market/reference data ingestion
  jobs/                RQ task definitions and orchestration
  models/              Model training, inference, and artifacts
  research/            Research workflow helpers
  services/            Application service layer
tests/                 Pytest test suite
```

## Quick start

Run these tasks in order:

1. Install `uv` if it is not already available: <https://docs.astral.sh/uv/>.
2. Copy environment defaults: `cp .env.example .env`.
3. Fill only local values in `.env`; do not commit secrets.
4. Install dependencies: `uv sync --dev`.
5. Run checks with timeouts: `uv run pytest --timeout=30`.
6. Start local services: `docker compose up --build`.
7. Open the FastAPI docs at <http://localhost:8000/docs>.
8. Open the Streamlit UI at <http://localhost:8501>.

## Local commands

```bash
uv run uvicorn apps.api.main:app --reload
uv run streamlit run apps/ui/Home.py
uv run pytest --timeout=30
uv run ruff check .
uv run mypy src
```

## Environment

`.env.example` documents required local variables and intentionally includes only blank or non-secret development defaults. `MASSIVE_API_KEY=` is left empty by design.
