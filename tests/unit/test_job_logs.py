"""Tests for user-visible background job logs."""

from __future__ import annotations

from quant_platform.common.enums import JobStatus
from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.jobs.queue import (
    append_job_log,
    list_job_logs,
    list_jobs_by_status,
    reconcile_stale_jobs,
)


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


class _FakeRegistry:
    def __init__(self, job_ids: set[str]) -> None:
        self._job_ids = job_ids

    def get_job_ids(self) -> list[str]:
        return sorted(self._job_ids)


class _FakeQueue:
    def __init__(self, registry_job_ids: set[str] | None = None) -> None:
        ids = registry_job_ids or set()
        self.queued_job_registry = _FakeRegistry(ids)
        self.started_job_registry = _FakeRegistry(set())
        self.deferred_job_registry = _FakeRegistry(set())
        self.scheduled_job_registry = _FakeRegistry(set())
        self.failed_job_registry = _FakeRegistry(set())
        self.finished_job_registry = _FakeRegistry(set())

    def fetch_job(self, job_id: str) -> object | None:
        return None


def test_reconcile_stale_jobs_marks_missing_rq_id_failed(tmp_path) -> None:
    """Jobs that never received an RQ id should fail during reconciliation."""

    from sqlalchemy import select

    from quant_platform.data.storage.catalog import jobs
    from quant_platform.jobs.queue import reconcile_stale_jobs

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    job_id = catalog.insert_row(
        "jobs",
        {"job_type": "training", "status": JobStatus.QUEUED.value, "payload": {}},
    )

    result = reconcile_stale_jobs(catalog=catalog, queue=_FakeQueue())

    assert result.checked == 1
    assert [item.catalog_job_id for item in result.reconciled] == [job_id]
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(select(jobs).where(jobs.c.id == job_id))
            .mappings()
            .one()
        )
    assert row["status"] == JobStatus.FAILED.value
    assert row["error"] == "missing rq_job_id; job was never enqueued"
    assert (
        list_job_logs(job_id, catalog=catalog)[-1]["metadata"]["action"]
        == "missing_rq_job_id"
    )


def test_reconcile_stale_jobs_cancels_missing_rq_job_and_experiment(tmp_path) -> None:
    """Stale RQ ids should be terminal in both jobs and linked experiments."""

    from sqlalchemy import select

    from quant_platform.data.storage.catalog import experiments, jobs
    from quant_platform.jobs.queue import reconcile_stale_jobs

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    experiment_id = catalog.insert_row(
        "experiments",
        {"name": "stale-exp", "status": JobStatus.RUNNING.value, "metadata": {}},
    )
    job_id = catalog.insert_row(
        "jobs",
        {
            "job_type": "training",
            "status": JobStatus.RUNNING.value,
            "payload": {"rq_job_id": "missing-rq", "experiment_id": experiment_id},
        },
    )

    result = reconcile_stale_jobs(catalog=catalog, queue=_FakeQueue())

    assert result.checked == 1
    assert result.reconciled[0].action == "stale_rq_job_missing"
    with catalog.engine.connect() as connection:
        job = (
            connection.execute(select(jobs).where(jobs.c.id == job_id))
            .mappings()
            .one()
        )
        experiment = connection.execute(
            select(experiments).where(experiments.c.id == experiment_id)
        ).mappings().one()
    assert job["status"] == JobStatus.CANCELLED.value
    assert experiment["status"] == JobStatus.CANCELLED.value
    assert list_job_logs(job_id, catalog=catalog)[-1]["message"].startswith(
        "RQ job not found"
    )


def test_reconcile_stale_jobs_treats_legacy_train_as_training(tmp_path) -> None:
    """Legacy train rows should still reconcile linked training experiments."""

    from sqlalchemy import select

    from quant_platform.data.storage.catalog import experiments

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    experiment_id = catalog.insert_row(
        "experiments",
        {"name": "legacy-exp", "status": JobStatus.QUEUED.value, "metadata": {}},
    )
    catalog.insert_row(
        "jobs",
        {
            "job_type": "train",
            "status": JobStatus.QUEUED.value,
            "payload": {"rq_job_id": "missing-rq", "experiment_id": experiment_id},
        },
    )

    result = reconcile_stale_jobs(catalog=catalog, queue=_FakeQueue())

    with catalog.engine.connect() as connection:
        experiment = connection.execute(
            select(experiments).where(experiments.c.id == experiment_id)
        ).mappings().one()
    assert result.reconciled[0].status == JobStatus.CANCELLED.value
    assert experiment["status"] == JobStatus.CANCELLED.value


def test_reconcile_stale_jobs_keeps_jobs_found_in_registry(tmp_path) -> None:
    """Jobs present in an RQ registry should not be changed."""

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    catalog.insert_row(
        "jobs",
        {
            "job_type": "training",
            "status": JobStatus.QUEUED.value,
            "payload": {"rq_job_id": "live-rq"},
        },
    )

    result = reconcile_stale_jobs(catalog=catalog, queue=_FakeQueue({"live-rq"}))

    assert result.checked == 1
    assert result.reconciled == []
    assert catalog.list_rows("jobs")[0]["status"] == JobStatus.QUEUED.value
