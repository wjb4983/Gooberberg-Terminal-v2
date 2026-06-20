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
tests/                 Pytest test suite
```

## Access assumptions

Use VSCode Remote - SSH to connect to the Ubuntu server over its Tailscale hostname or Tailscale IP. Run the commands below in the VSCode remote terminal from the repository root. Forward ports `8000` and `8501` in the VSCode **Ports** panel if you want to open the API docs or Streamlit UI in your laptop browser.

## First-time start after cloning

Run these commands in this order:

```bash
cd /workspace/Gooberberg-Terminal-v2
cp .env.example .env
# Edit .env and set MASSIVE_API_KEY if you want live Massive.com data.
# Leave MASSIVE_API_KEY blank if you only want fake/mock local data.
uv sync --dev
uv run python scripts/init_metadata_db.py
docker compose up -d redis
uv run pytest tests --timeout=60
```

Then start the three app processes in three separate VSCode terminals:

```bash
cd /workspace/Gooberberg-Terminal-v2
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
cd /workspace/Gooberberg-Terminal-v2
uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501
```

```bash
cd /workspace/Gooberberg-Terminal-v2
uv run python -m quant_platform.jobs.workers
```

Open:

- FastAPI docs: <http://localhost:8000/docs>
- Streamlit UI: <http://localhost:8501>

If you are using VSCode SSH from your laptop, forward ports `8000` and `8501` when VSCode prompts you, or open them manually from the **Ports** panel.

## Normal daily start

If the repo is already set up and you just want to run it:

```bash
cd /workspace/Gooberberg-Terminal-v2
docker compose up -d redis
```

Start these in three separate terminals:

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
```

```bash
uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501
```

```bash
uv run python -m quant_platform.jobs.workers
```

## After pulling new changes

Run this after every `git pull`:

```bash
cd /workspace/Gooberberg-Terminal-v2
git pull
uv sync --dev
uv run python scripts/init_metadata_db.py
docker compose up -d redis
uv run pytest tests --timeout=60
```

Then restart the API, UI, and worker terminals if they were already running.

## Useful commands

```bash
# Run the full test suite with a timeout
uv run pytest tests --timeout=60

# Run only unit tests with a timeout
uv run pytest tests/unit --timeout=60

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run mypy src

# Stop Redis and other compose services
docker compose down
```

## Environment and secrets

Create a local environment file from the example:

```bash
cp .env.example .env
```

Put secrets only in `.env` or your shell environment. Never commit `.env`, API keys, tokens, credentials, or other secrets.

Common local values:

```bash
MASSIVE_API_KEY=your_live_key_here
QUANT_PLATFORM_REDIS_URL=redis://localhost:6379/0
QUANT_PLATFORM_DATA_LAKE_ROOT=./data/lake
QUANT_PLATFORM_CATALOG_DB_PATH=./data/catalog/catalog.db
QUANT_PLATFORM_API_BASE_URL=http://localhost:8000
```

Before committing, run:

```bash
git status --short
```

Make sure `.env` is not staged.

## More documentation

- [Architecture](docs/architecture.md)
- [Data lake](docs/data_lake.md)
- [HPC setup](docs/hpc_setup.md)
- [Massive provider](docs/massive_provider.md)
