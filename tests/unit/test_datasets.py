"""Tests for logical dataset modules."""

from __future__ import annotations

import json

from quant_platform.common.enums import DataType, Provider
from quant_platform.datasets.materialization import DatasetMaterializer
from quant_platform.datasets.registry import DatasetRegistry
from quant_platform.datasets.schemas import DatasetDefinition


def test_dataset_registry_persists_logical_definition(tmp_path):
    db_path = tmp_path / "metadata.sqlite"
    config_dir = tmp_path / "configs" / "datasets"
    registry = DatasetRegistry(db_path, config_dir=config_dir)
    definition = DatasetDefinition(
        name="us_equity_daily",
        version="2026-01",
        asset_universe=["SPY", "QQQ"],
        provider=Provider.MASSIVE,
        data_types=[DataType.DAILY_BARS],
        resolution="1d",
        date_range={"start": "2024-01-01", "end": "2024-12-31"},
        filters={"adjusted": True},
        feature_set={"name": "daily_alpha", "version": "1"},
    )

    dataset_id = registry.register(definition, mirror_config=True)

    assert dataset_id > 0
    loaded = registry.get("us_equity_daily")
    assert loaded == definition
    mirrored = json.loads((config_dir / "us_equity_daily.json").read_text())
    assert mirrored["source"] == "logical_view"
    assert "storage_uri" not in mirrored


def test_materializer_builds_logical_plan_without_artifact(tmp_path):
    registry = DatasetRegistry(tmp_path / "metadata.sqlite", config_dir=None)
    definition = DatasetDefinition(
        name="crypto_ticks",
        asset_universe=["BTC-USD"],
        provider=Provider.INTERNAL,
        data_types=[DataType.TRADES, DataType.QUOTES],
        resolution="tick",
    )
    registry.register(definition)

    plan = DatasetMaterializer(registry).build_plan("crypto_ticks")

    assert plan.logical is True
    assert plan.artifact_uri is None
    assert plan.query["provider"] == "internal"
    assert plan.query["data_types"] == ["trades", "quotes"]
