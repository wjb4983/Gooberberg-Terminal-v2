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


def test_context_helpers_filter_regime_switching_options() -> None:
    from quant_platform.datasets.console import (
        data_type_options_for_asset_class,
        default_resolution_for_context,
        default_symbols_for_context,
        provider_options_for_asset_class,
    )

    assert provider_options_for_asset_class("equity") == [
        "massive",
        "polygon",
        "alpaca",
        "yahoo",
    ]
    assert data_type_options_for_asset_class("equity", "learned_regime_switching") == [
        "daily_bars",
        "bars",
    ]
    assert "news" not in data_type_options_for_asset_class(
        "equity", "learned_regime_switching"
    )
    assert default_resolution_for_context("equity", "learned_regime_switching") == "1d"
    assert default_symbols_for_context("equity", "learned_regime_switching")[:3] == [
        "SPY",
        "QQQ",
        "IWM",
    ]


def test_build_definition_persists_workflow_metadata() -> None:
    definition = build_definition(
        name="regime_dataset",
        version="1",
        provider="massive",
        asset_universe=["SPY", "QQQ"],
        data_types=["daily_bars"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        asset_class="equity",
        resolution="1d",
        workflow_intent="learned_regime_switching",
        regime_source="real_market_data",
        target_style="regime_labels",
        preferred_task_type="regime_classification",
        feature_requirements=["returns", "volatility", "trend"],
        labeling_intent="Learned regime switching",
    )

    assert definition.metadata["workflow_intent"] == "learned_regime_switching"
    assert definition.metadata["asset_class"] == "equity"
    assert definition.metadata["labeling_intent"] == "Learned regime switching"
    assert definition.metadata["regime_source"] == "real_market_data"
    assert definition.metadata["target_style"] == "regime_labels"
    assert definition.metadata["preferred_task_type"] == "regime_classification"
    assert definition.metadata["feature_requirements"] == [
        "returns",
        "volatility",
        "trend",
    ]


def test_build_definition_defaults_regime_compatibility_metadata() -> None:
    definition = build_definition(
        name="regime_dataset",
        version="1",
        provider="massive",
        asset_universe=["SPY"],
        data_types=["daily_bars"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        asset_class="equity",
        resolution="1d",
        workflow_intent="learned_regime_switching",
    )

    assert definition.metadata == {
        "asset_class": "equity",
        "workflow_intent": "learned_regime_switching",
        "regime_source": "real_market_data",
        "preferred_task_type": "regime_classification",
    }


def test_build_definition_explicit_metadata_fields_override_extra_metadata() -> None:
    definition = build_definition(
        name="regime_dataset",
        version="1",
        provider="massive",
        asset_universe=["SPY"],
        data_types=["daily_bars"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        asset_class="equity",
        resolution="1d",
        workflow_intent="learned_regime_switching",
        preferred_task_type="regime_classification",
        metadata={
            "asset_class": "crypto",
            "workflow_intent": "raw_market_data",
            "preferred_task_type": "forecasting",
            "custom": "kept",
        },
    )

    assert definition.metadata["asset_class"] == "equity"
    assert definition.metadata["workflow_intent"] == "learned_regime_switching"
    assert definition.metadata["preferred_task_type"] == "regime_classification"
    assert definition.metadata["custom"] == "kept"
