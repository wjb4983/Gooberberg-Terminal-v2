# Experiment queueing

This guide documents the local startup order for experiment work, how training experiments are queued, and the API endpoints to inspect queue and experiment status.

## Recommended local startup order

Run these quick-start tasks from the repository root in this order:

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` if you need live market data or non-default local paths. For local mock workflows, `MASSIVE_API_KEY` can stay blank.

2. Install development dependencies:

   ```bash
   uv sync --dev
   ```

3. Run the test suite with a timeout:

   ```bash
   uv run pytest tests --timeout=60
   ```

4. Start the local development stack:

   ```bash
   uv run gooberberg-dev
   ```

   This command initializes metadata storage, starts Redis through Docker Compose, and runs FastAPI, Streamlit, and the RQ worker together for local development.

5. Open the local browser surfaces:

   - FastAPI docs: <http://localhost:8000/docs>
   - Streamlit UI: <http://localhost:8501>

   If you are connected through VSCode Remote - SSH, forward ports `8000` and `8501` from the VSCode **Ports** panel before opening them in your laptop browser.

6. Queue a small experiment from FastAPI docs, Streamlit, or the API. Keep the first run intentionally small, for example two epochs and a short date range, so you can verify the queue quickly before submitting larger training work.

7. Check the local job board:

   ```bash
   curl http://localhost:8000/api/v1/jobs/board
   ```

   Confirm the new job appears in the queued, running, or finished groups.

8. Check job logs for the queued experiment:

   ```bash
   curl http://localhost:8000/api/v1/jobs/{job_id}/logs
   ```

   Replace `{job_id}` with the metadata `job_id` returned when you queued the experiment.

9. Cancel/delete stale queued or running jobs if needed:

   ```bash
   curl -X DELETE http://localhost:8000/api/v1/jobs/{job_id}
   ```

   Use this only for work you no longer want to run. The endpoint cancels or deletes the Redis/RQ job when possible and marks the catalog job `cancelled` so stale work no longer appears as active.

## How experiment jobs are queued

`POST /api/v1/experiments` creates metadata rows before work is processed by the background worker:

- A row in `experiments` stores the experiment name, queued status, model and dataset references, parameters, metadata, and queued payload.
- A row in `jobs` stores the background training job, its status, payload, result, error, and lifecycle timestamps.
- Rows in `job_logs` store user-visible lifecycle messages for the job as the RQ worker progresses.

Redis/RQ handles execution. Redis is the queue broker, and the RQ worker consumes the queued training payload. The metadata catalog remains the source of local observability for the API and UI: use the `jobs`, `job_logs`, and `experiments` rows to inspect what was submitted, what is running, and what completed.

## Queue an experiment

The experiment endpoint expects existing dataset and model definition IDs. Supervised neural-network training is the supported queueing path today.

Example request:

```bash
curl -X POST http://localhost:8000/api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "demo-supervised-training",
    "dataset_id": 1,
    "feature_set_id": null,
    "model_id": 1,
    "experiment_kind": "supervised_training",
    "task_type": "regression",
    "target": {
      "column": "close",
      "horizon": 1
    },
    "split": {
      "train_start": "2024-01-01",
      "train_end": "2024-03-31",
      "validation_start": "2024-04-01",
      "validation_end": "2024-04-30",
      "test_start": "2024-05-01",
      "test_end": "2024-05-31"
    },
    "training": {
      "batch_size": 16,
      "epochs": 2,
      "optimizer": "adam",
      "learning_rate": 0.001,
      "loss_function": "mse",
      "sequence_length": 8,
      "hidden_size": 16,
      "seed": 7,
      "synthetic_rows_per_day": 4
    },
    "metadata": {
      "owner": "local-dev",
      "purpose": "queue smoke test"
    }
  }'
```

A successful response includes the new `experiment_id`, the metadata `job_id`, the job `status`, and the normalized training payload.

## Inspect queue and experiment status

Use these endpoints after queueing an experiment:

```bash
curl http://localhost:8000/api/v1/jobs/board
```

Returns queued, running, and finished job groups from the `jobs` metadata table. Use this as the local job board.

```bash
curl http://localhost:8000/api/v1/jobs/{job_id}/logs
```

Returns chronological user-visible log rows from `job_logs` for one background job. Replace `{job_id}` with the `job_id` returned by `POST /api/v1/experiments`.

```bash
curl http://localhost:8000/api/v1/experiments/{experiment_id}
```

Returns the experiment row, parameters, metadata, metrics, artifact links, and timestamps. Replace `{experiment_id}` with the `experiment_id` returned by the queue request.

## Reconcile stale queued/running jobs

Use reconciliation when the metadata catalog shows jobs stuck in `queued` or `running` but Redis/RQ was restarted, flushed, or otherwise no longer has matching job state. The cleanup checks every queued/running catalog job against the RQ queue plus the queued, started, deferred, scheduled, failed, and finished registries.

Recommended cleanup workflow:

1. Inspect the job board and identify stuck work:

   ```bash
   curl http://localhost:8000/api/v1/jobs/board
   ```

2. Reconcile the catalog against RQ state from the API:

   ```bash
   curl -X POST http://localhost:8000/api/v1/jobs/reconcile
   ```

   Or run the same maintenance task from the command line:

   ```bash
   python -m quant_platform.jobs.reconcile
   ```

3. Review the returned `checked` count and `reconciled` list, then inspect logs for any changed jobs:

   ```bash
   curl http://localhost:8000/api/v1/jobs/{job_id}/logs
   ```

4. Re-submit any experiment that should run again after reconciliation.

Reconciliation marks jobs with no `payload.rq_job_id` as `failed` with the error `missing rq_job_id; job was never enqueued`. Jobs with an `rq_job_id` that is absent from all inspected RQ states are marked `cancelled` by default and receive a warning log entry. If a reconciled job payload has `experiment_id`, the linked experiment is moved from `queued` or `running` to the same terminal status.

## Troubleshooting: job remains queued

If a newly submitted experiment stays in `queued` longer than expected, check the queue dependencies in this order:

1. Confirm Redis is running. `uv run gooberberg-dev` starts Redis through Docker Compose for the normal local workflow; if you started services manually, verify the Redis container or process is healthy before re-submitting work.
2. Confirm the RQ worker process is running. The worker is the process that consumes queued training jobs from Redis and updates catalog rows from `queued` to `running`, then to a terminal status.
3. Confirm the catalog job payload has an `rq_job_id`. Check the job from `/api/v1/jobs/board` or inspect the catalog row directly. Current RQ-backed submissions store the Redis/RQ identifier in `payload.rq_job_id`; without it, the API can show metadata for the job but cannot match it to a Redis/RQ job.

For stale queued or running jobs that you do not want to keep, call the cancel/delete endpoint:

```bash
curl -X DELETE http://localhost:8000/api/v1/jobs/{job_id}
```

If you want the catalog to reconcile all active rows against Redis/RQ state instead, call:

```bash
curl -X POST http://localhost:8000/api/v1/jobs/reconcile
```

## Old metadata-only jobs from before RQ integration

Some local catalogs may contain old metadata-only jobs created before Redis/RQ queueing was integrated. Those rows may have `queued` or `running` status but no `payload.rq_job_id`, because no Redis/RQ job was ever created for them.

These legacy rows are observable catalog records only; the worker cannot pick them up from Redis, and the cancel/delete endpoint has no RQ job to remove. When you delete one of these jobs with `DELETE /api/v1/jobs/{job_id}`, the catalog row is still marked `cancelled` and a warning is recorded so it no longer appears as active work. When you run reconciliation, jobs missing `payload.rq_job_id` are marked `failed` with the error `missing rq_job_id; job was never enqueued`.
