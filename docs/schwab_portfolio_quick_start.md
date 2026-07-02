# Schwab Portfolio Page Quick-Start Tasks

Use these tasks in order to add a Schwab holdings page that displays current positions and calculates Sharpe ratio, beta, max drawdown, volatility, and related portfolio metrics across multiple lookback windows.

## Task 1: Confirm Schwab app access and redirect configuration

1. Confirm your Schwab Developer app has the account/position and market-data access needed for holdings, quotes, and historical prices.
2. Choose the local redirect URI you will use for OAuth bootstrap, for example `https://127.0.0.1:8182/callback` or another URI allowed by Schwab.
3. Record only non-secret setup notes in project docs; never commit Schwab client secrets, refresh tokens, access tokens, account numbers, or raw authorization responses.

## Task 2: Add Schwab configuration settings

1. Add Schwab settings to `src/quant_platform/config/settings.py` for client ID, client secret, redirect URI, token path, API timeout, default benchmark, and supported lookback windows.
2. Keep secret fields out of `/settings` responses and logs.
3. Document the new environment variables in `.env.example` or the README with placeholder values only.
4. Add config tests for default values and environment overrides.
5. Run `uv run pytest tests/unit/test_config_validation.py --timeout=60`.

## Task 3: Add a Schwab OAuth bootstrap command

1. Add a CLI helper that builds the Schwab authorization URL from settings.
2. Let the user paste the redirected callback URL or authorization code into the terminal.
3. Exchange the code for tokens and write the token file to the configured token path with restrictive permissions.
4. Never print or log the access token, refresh token, client secret, or unmasked account identifiers.
5. Add mocked tests for authorization URL construction, callback parsing, token exchange, and token-file writes.
6. Run `uv run pytest tests/unit/test_schwab_auth.py --timeout=60`.

## Task 4: Build a Schwab provider adapter with mocked tests first

1. Add `src/quant_platform/data/providers/schwab.py`.
2. Create a small adapter for account-number lookup, account holdings, quotes, and historical price data.
3. Normalize Schwab holdings into internal fields such as account hash, masked account label, symbol, asset type, quantity, market value, current price, cost basis, average price, unrealized PnL, and unrealized PnL percent.
4. Add fake/mocked transport support so unit tests never call live Schwab APIs.
5. Add tests for payload normalization, missing fields, token refresh behavior, HTTP timeout configuration, and account masking.
6. Run `uv run pytest tests/unit/test_schwab_provider.py --timeout=60`.

## Task 5: Add reusable portfolio metric functions

1. Add `src/quant_platform/portfolio/metrics.py`.
2. Implement portfolio weights, daily portfolio returns, annualized volatility, Sharpe ratio, beta versus benchmark, max drawdown, total return, and lookback slicing.
3. Support lookbacks in this initial order: `1M`, `3M`, `6M`, `YTD`, `1Y`, `3Y`, and `MAX`.
4. Define behavior for cash-only portfolios, missing prices, single-day histories, zero-volatility returns, and benchmark alignment.
5. Add deterministic tests with small in-memory price series.
6. Run `uv run pytest tests/unit/test_portfolio_metrics.py --timeout=60`.

## Task 6: Add a portfolio analytics service

1. Add `src/quant_platform/portfolio/service.py`.
2. Fetch Schwab holdings, price histories, and benchmark history through the provider adapter.
3. Compute account totals, allocation by symbol, allocation by asset type, holdings rows, and lookback metrics.
4. Return warnings for partial data problems instead of failing the whole response when practical.
5. Include refresh timestamps and metadata that the UI can display.
6. Add mocked service tests for normal holdings, partial missing price history, empty holdings, cash-only holdings, and benchmark failures.
7. Run `uv run pytest tests/unit/test_portfolio_service.py --timeout=60`.

## Task 7: Add FastAPI portfolio routes

1. Add `apps/api/routes/portfolio.py` with response models for holdings, allocations, metrics, and warnings.
2. Add endpoints such as `GET /api/v1/portfolio/holdings` and `GET /api/v1/portfolio/metrics`.
3. Register the router in `apps/api/main.py`.
4. Ensure route responses never include secrets, raw tokens, full account numbers, or raw Schwab authorization payloads.
5. Add mocked API tests for success, missing Schwab configuration, empty holdings, and service warnings.
6. Run `uv run pytest tests/unit/test_portfolio_routes.py --timeout=60`.

## Task 8: Add the Streamlit Portfolio page

1. Add `apps/ui/pages/6_Portfolio.py`.
2. Add controls in this order: account selector, benchmark selector, lookback selector, and refresh action.
3. Display total market value, cash value when available, unrealized PnL, allocation charts, holdings table, and a lookback metrics table.
4. Show clear setup guidance when Schwab settings or token files are missing.
5. Mask account identifiers everywhere on the page.
6. Keep Streamlit session state free of secrets and token contents.
7. Run `uv run pytest tests/unit --timeout=60`.

## Task 9: Add caching and rate-limit protection

1. Cache holdings for a short TTL, such as 30 to 120 seconds.
2. Cache historical daily price data by symbol and date range for longer, such as through end-of-day or 24 hours.
3. Add explicit HTTP timeouts to all Schwab network calls.
4. Surface last-refresh timestamps and stale-data warnings to API and UI consumers.
5. Add cache tests to verify repeated service calls do not repeatedly invoke the mocked Schwab transport.
6. Run `uv run pytest tests/unit/test_portfolio_service.py tests/unit/test_schwab_provider.py --timeout=60`.

## Task 10: Run final checks before opening the pull request

1. Run `uv run ruff format .`.
2. Run `uv run ruff check .`.
3. Run `uv run mypy src`.
4. Run `uv run pytest tests --timeout=60`.
5. Start the app with `uv run gooberberg-dev`, open Streamlit, and verify the Portfolio page behavior with mocked or real Schwab credentials.
6. If the UI changes are visible, capture a screenshot for the pull request.
7. Run `git status --short` and confirm no `.env`, token files, credentials, screenshots with sensitive account data, or other secrets are staged.
