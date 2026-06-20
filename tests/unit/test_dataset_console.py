"""Tests for dataset page orchestration helpers."""

from __future__ import annotations

from datetime import date

from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.datasets.console import (
    build_definition,
    checks_passed,
    parse_asset_universe,
    preview_coverage,
    queue_ingestion,
    validate_definition_inputs,
)


def test_parse_asset_universe_normalizes_and_deduplicates_symbols() -> None:
    assert parse_asset_universe(" aapl, MSFT\naapl, spy ") == ["AAPL", "MSFT", "SPY"]


def test_validation_checks_report_invalid_inputs() -> None:
    checks = validate_definition_inputs(
        name="",
        asset_universe=[],
        data_types=[],
        start=date(2025, 1, 2),
        end=date(2025, 1, 1),
        provider="massive",
        asset_class="equity",
    )

    assert not checks_passed(checks)
    assert [check.name for check in checks if not check.passed] == [
        "Dataset name",
        "Asset universe",
        "Data types",
        "Date range",
    ]


def test_preview_coverage_returns_missing_plan(tmp_path) -> None:
    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    definition = build_definition(
        name="equity_daily_bars",
        version="1",
        provider="massive",
        asset_universe=["AAPL", "MSFT"],
        data_types=["daily_bars"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        asset_class="equity",
        resolution="1d",
    )

    preview = preview_coverage(definition, catalog)

    assert preview.requested_count == 4
    assert preview.missing_count == 4
    assert preview.rows[0]["status"] == "missing"
    assert preview.rows[0]["dataset"] == "market_data.equity.daily_bars"


def test_queue_ingestion_registers_dataset_and_job(tmp_path) -> None:
    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    definition = build_definition(
        name="equity_daily_bars",
        version="1",
        provider="massive",
        asset_universe=["AAPL"],
        data_types=["daily_bars"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        asset_class="equity",
        resolution="1d",
    )

    queued = queue_ingestion(
        definition,
        catalog=catalog,
        config_dir=tmp_path / "configs" / "datasets",
    )

    assert queued.dataset_id == 1
    assert queued.job_id == 1
    assert queued.status == "queued"
    assert queued.payload["symbols"] == ["AAPL"]
    assert queued.payload["dataset_name"] == "equity_daily_bars"
