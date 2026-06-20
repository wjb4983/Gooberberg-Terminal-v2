"""Tests for monitoring summary placeholders and catalog signals."""

from __future__ import annotations

from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.monitoring.service import MonitoringService


def test_monitoring_summary_labels_unimplemented_live_trading(tmp_path) -> None:
    """Monitoring should explicitly show unimplemented deployment/trading pieces."""

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    catalog.insert_row(
        "model_definitions",
        {
            "name": "demo-model",
            "version": "1",
            "model_type": "mlp",
            "artifact_uri": "models/demo.pt",
            "parameters": {},
            "metadata": {},
        },
    )
    catalog.insert_row(
        "jobs",
        {"job_type": "training", "status": "queued", "payload": {"model": "demo"}},
    )

    summary = MonitoringService(catalog).summary().jsonable()

    assert summary["active_models"][0]["deployment_status"] == "not_implemented"
    assert summary["active_models"][0]["serving_status"] == "not_implemented"
    assert summary["recent_predictions"][0]["status"] == "not_implemented"
    assert summary["drift_checks"][0]["status"] == "not_implemented"
    assert summary["latency"]["queued_or_running_jobs"] == 1
    assert summary["trading_status"]["live_trading"]["status"] == "not_implemented"
    assert summary["trading_status"]["paper_trading"]["status"] == "not_implemented"
