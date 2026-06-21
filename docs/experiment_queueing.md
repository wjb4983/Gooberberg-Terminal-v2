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
