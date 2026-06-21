# Massive provider

The Massive.com provider is the platform's live market-data integration. It should be used only with a valid local API key and in compliance with your Massive.com plan and data-license terms.

## Secrets and `.env` setup

Create local configuration:

```bash
cp .env.example .env
```

Set the Massive API key locally:

```bash
MASSIVE_API_KEY=your_live_key_here
```

You may also set supporting platform values:

```bash
QUANT_PLATFORM_REDIS_URL=redis://localhost:6379/0
QUANT_PLATFORM_DATA_LAKE_ROOT=./data/lake
QUANT_PLATFORM_CATALOG_DB_PATH=./data/catalog/catalog.db
QUANT_PLATFORM_API_BASE_URL=http://localhost:8000
```

Never commit `MASSIVE_API_KEY`, `.env`, shell history containing secrets, downloaded proprietary datasets, or vendor credentials.

## Access assumptions

When developing remotely, connect to the host with VSCode Remote - SSH over Tailscale. Tailscale is assumed to provide private connectivity to the development host; this project does not publish Massive-backed endpoints to the public internet. Keep API and UI ports private or forwarded through SSH/Tailscale.

## Recommended workflow order

1. Connect to the remote host with VSCode Remote - SSH over Tailscale.
2. Open the repository in the remote VSCode window.
3. Copy `.env.example` to `.env`.
4. Add `MASSIVE_API_KEY` to `.env` or export it in your shell.
5. Check that `.env` is ignored and not staged.
6. Install dependencies: `uv sync --dev`.
7. Run provider tests with a timeout: `uv run pytest tests/unit/test_massive_provider.py --timeout=60`.
8. Start the full local stack with `uv run gooberberg-dev`.


## Direct Python execution

Run services directly during development:

```bash
uv run gooberberg-dev
```

## Optional Docker Compose services

Use Compose for Redis or full local service containers when useful:

```bash
docker compose up --build
```

For the direct Python workflow, `uv run gooberberg-dev` starts Redis automatically unless you pass `--skip-redis`.

## Tests with timeouts

Use explicit test timeouts:

```bash
uv run pytest tests/unit/test_massive_provider.py --timeout=60
uv run pytest tests/unit/test_ingestion.py --timeout=60
uv run pytest --timeout=60
```

Provider tests should avoid making live network calls unless they are explicitly marked and configured for that purpose. Keep unit tests deterministic with fake or mocked provider responses.
