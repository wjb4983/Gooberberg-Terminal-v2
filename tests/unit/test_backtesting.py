from __future__ import annotations

import json

import pandas as pd

from quant_platform.backtesting import (
    BacktestConfig,
    run_backtest,
    write_backtest_artifacts,
)


def test_run_backtest_computes_metrics_and_trades() -> None:
    market_data = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "symbol": "AAPL",
                "close": 100.0,
                "signal": 1.0,
            },
            {
                "timestamp": "2024-01-02",
                "symbol": "AAPL",
                "close": 110.0,
                "signal": 0.0,
            },
        ]
    )
    config = BacktestConfig(initial_cash=100_000.0)

    result = run_backtest(market_data, config=config)

    assert result.metrics["number_of_trades"] == 2
    assert result.metrics["total_return"] > 0
    assert len(result.equity_curve) == 2
    assert len(result.trades) == 2


def test_write_backtest_artifacts(tmp_path) -> None:
    market_data = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01",
                "symbol": "MSFT",
                "close": 50.0,
                "signal": 1.0,
            },
            {
                "timestamp": "2024-01-02",
                "symbol": "MSFT",
                "close": 55.0,
                "signal": 0.0,
            },
        ]
    )
    result = run_backtest(market_data, config=BacktestConfig(name="artifact-test"))

    manifest = write_backtest_artifacts(result, artifact_dir=tmp_path)

    assert (tmp_path / "equity_curve.parquet").exists()
    assert (tmp_path / "trades.parquet").exists()
    assert json.loads((tmp_path / "metrics.json").read_text())["number_of_trades"] == 2
    assert json.loads((tmp_path / "config.json").read_text())["name"] == "artifact-test"
    assert manifest.files["config"].endswith("config.json")
