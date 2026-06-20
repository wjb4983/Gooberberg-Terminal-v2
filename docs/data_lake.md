# Data lake

The local data lake stores market-data artifacts, generated datasets, manifests, and catalog metadata for development and research workflows. It is configured through environment-backed settings and is intended to be reproducible from provider downloads and processing jobs rather than committed to Git.

## Paths and configuration

Create local configuration first:

```bash
cp .env.example .env
```

Recommended local values:

```bash
MASSIVE_API_KEY=your_live_key_here
QUANT_PLATFORM_DATA_LAKE_ROOT=./data/lake
QUANT_PLATFORM_CATALOG_DB_PATH=./data/catalog/catalog.db
QUANT_PLATFORM_REDIS_URL=redis://localhost:6379/0
QUANT_PLATFORM_API_BASE_URL=http://localhost:8000
```

`MASSIVE_API_KEY` is required only for live Massive.com calls. Do not commit `.env`, provider keys, data-vendor credentials, downloaded proprietary data, or other secrets.

## Data-lake layout

The data-lake root is controlled by `QUANT_PLATFORM_DATA_LAKE_ROOT`. Ingestion services write partitioned artifacts under that root, while ingestion manifests and coverage metadata are recorded in the catalog path configured by `QUANT_PLATFORM_CATALOG_DB_PATH`.

Typical local development layout:

```text
data/
  lake/        Parquet artifacts and partitioned market data
  catalog/     SQLite metadata catalog and ingestion manifests
```

## Recommended setup order

1. Connect with VSCode Remote - SSH over Tailscale.
2. Open the repository on the remote development host.
3. Copy `.env.example` to `.env`.
4. Add `MASSIVE_API_KEY` if live market-data ingestion is required.
5. Verify `.env` is ignored and not staged.
6. Install dependencies: `uv sync --dev`.
7. Run tests with a timeout: `uv run pytest tests/unit --timeout=30`.
8. Start Redis if queue-backed ingestion jobs are needed: `docker compose up redis`.
9. Start a worker: `uv run python -m quant_platform.jobs.workers`.
10. Start the API: `uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload`.

## Direct Python execution

Use direct commands for services that interact with data-lake state:

```bash
uv run python scripts/init_metadata_db.py
uv run python -m quant_platform.jobs.workers
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501
```

## Optional Docker Compose services

Redis can be run with Compose while Python processes run directly:

```bash
docker compose up redis
```

The API and UI container services are also available when needed:

```bash
docker compose up api
docker compose up ui
docker compose up --build
```

## Tests and validation

Run tests with explicit timeouts:

```bash
uv run pytest tests/unit/test_datasets.py --timeout=30
uv run pytest tests/unit/test_ingestion.py --timeout=30
uv run pytest tests/unit/test_storage_paths.py --timeout=30
uv run pytest --timeout=30
```
