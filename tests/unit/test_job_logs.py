"""Tests for user-visible background job logs."""

from __future__ import annotations

from quant_platform.common.enums import JobStatus
from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.jobs.queue import append_job_log, list_job_logs, list_jobs_by_status


def test_job_logs_are_appended_and_listed_in_order(tmp_path) -> None:
    """Catalog-backed job logs should be available for monitoring views."""

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    job_id = catalog.insert_row(
        "jobs",
        {"job_type": "training", "status": JobStatus.QUEUED.value, "payload": {}},
    )

    append_job_log(job_id, "Queued training job.", catalog=catalog)
    append_job_log(
        job_id,
        "Worker picked up job.",
        metadata={"worker": "dev"},
        catalog=catalog,
    )

    logs = list_job_logs(job_id, catalog=catalog)

    assert [log["message"] for log in logs] == [
        "Queued training job.",
        "Worker picked up job.",
    ]
    assert logs[1]["metadata"] == {"worker": "dev"}


def test_jobs_can_be_filtered_by_monitoring_status(tmp_path) -> None:
    """Monitoring helpers should separate queued/running/finished jobs."""

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    queued_id = catalog.insert_row(
        "jobs",
        {"job_type": "ingest", "status": JobStatus.QUEUED.value, "payload": {}},
    )
    catalog.insert_row(
        "jobs",
        {"job_type": "train", "status": JobStatus.SUCCEEDED.value, "payload": {}},
    )

    queued = list_jobs_by_status({JobStatus.QUEUED.value}, catalog=catalog)

    assert [job["id"] for job in queued] == [queued_id]
