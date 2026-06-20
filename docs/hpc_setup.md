# HPC setup

This guide describes a safe development workflow for using Gooberberg Terminal v2 on a remote workstation, lab server, or HPC login/development node.

## Access assumptions

The expected access path is VSCode Remote - SSH over Tailscale. Tailscale should already be installed, authenticated, and authorized by your organization or tailnet owner. The repository assumes private Tailscale reachability and does not manage tailnet ACLs, cluster firewalls, SSH daemon configuration, or public ingress.

Do not expose development ports publicly from an HPC or shared server. Prefer SSH port forwarding, VSCode port forwarding, or Tailscale-only access.

## VSCode SSH workflow

1. Verify the remote host is online in Tailscale.
2. Verify SSH access to the remote host from your local machine.
3. In VSCode, install **Remote - SSH**.
4. Add a host entry that uses the Tailscale DNS name or 100.x IP.
5. Connect with **Remote-SSH: Connect to Host...**.
6. Open `/workspace/Gooberberg-Terminal-v2` or the path where the repository is checked out.
7. Use VSCode terminals on the remote host for all commands.
8. Forward ports `8000` and `8501` only when you need browser access.

## Recommended quick-start order

1. SSH into the remote host through VSCode over Tailscale.
2. Load any site-required modules for Python, compilers, or Docker if your environment requires them.
3. Install `uv` in your user environment if it is not already present.
4. Copy `.env.example` to `.env`.
5. Set `MASSIVE_API_KEY` in `.env` only if live Massive.com access is needed.
6. Confirm secrets are not staged for commit.
7. Install dependencies: `uv sync --dev`.
8. Run tests with a timeout: `uv run pytest --timeout=30`.
9. Start optional Redis with `docker compose up redis` if Docker is available and allowed.
10. Start the API directly: `uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload`.
11. Start Streamlit directly: `uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501`.
12. Start an RQ worker directly: `uv run python -m quant_platform.jobs.workers`.

## Environment and secrets

Create `.env` from the example:

```bash
cp .env.example .env
```

Use local values similar to:

```bash
MASSIVE_API_KEY=your_live_key_here
QUANT_PLATFORM_REDIS_URL=redis://localhost:6379/0
QUANT_PLATFORM_DATA_LAKE_ROOT=./data/lake
QUANT_PLATFORM_CATALOG_DB_PATH=./data/catalog/catalog.db
QUANT_PLATFORM_API_BASE_URL=http://localhost:8000
```

Never commit `.env`, `MASSIVE_API_KEY`, SSH keys, cluster tokens, scheduler credentials, notebook tokens, or downloaded vendor data.

## Direct Python execution

Direct execution is preferred on HPC development nodes because it avoids unnecessary container rebuilds and works well with VSCode terminals:

```bash
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload
uv run streamlit run apps/ui/Home.py --server.address 0.0.0.0 --server.port 8501
uv run python -m quant_platform.jobs.workers
```

If a scheduler is required for long jobs, wrap these commands in the site-approved scheduler mechanism rather than running on shared login nodes.

## Optional Docker Compose services

Only use Docker Compose where containers are permitted by the remote environment:

```bash
docker compose up redis
docker compose up api
docker compose up ui
docker compose up --build
```

On clusters where Docker is unavailable, run Redis through the site-supported service mechanism or point `QUANT_PLATFORM_REDIS_URL` at an approved Redis endpoint.

## Tests and checks with timeouts

Use explicit timeouts for tests:

```bash
uv run pytest --timeout=30
uv run pytest tests/unit --timeout=30
uv run ruff check .
uv run mypy src
```
