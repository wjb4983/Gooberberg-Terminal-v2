"""Shared enumerations used across the quant platform."""

from __future__ import annotations

from enum import StrEnum


class AssetClass(StrEnum):
    """Supported instrument asset classes."""

    EQUITY = "equity"
    ETF = "etf"
    FOREX = "forex"
    CRYPTO = "crypto"
    FUTURE = "future"
    OPTION = "option"
    INDEX = "index"


class DataType(StrEnum):
    """Supported market and reference data granularities."""

    TRADES = "trades"
    QUOTES = "quotes"
    BARS = "bars"
    DAILY_BARS = "daily_bars"
    FUNDAMENTALS = "fundamentals"
    CORPORATE_ACTIONS = "corporate_actions"
    NEWS = "news"


class Provider(StrEnum):
    """External data and execution providers."""

    MASSIVE = "massive"
    POLYGON = "polygon"
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    INTERNAL = "internal"


class TaskType(StrEnum):
    """Background task categories."""

    INGEST = "ingest"
    BACKFILL = "backfill"
    FEATURE_BUILD = "feature_build"
    TRAIN = "train"
    EVALUATE = "evaluate"
    PREDICT = "predict"
    EXPORT = "export"


class ModelType(StrEnum):
    """Model families supported by the platform."""

    BASELINE = "baseline"
    LINEAR = "linear"
    TREE = "tree"
    BOOSTING = "boosting"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"


class JobStatus(StrEnum):
    """Lifecycle states for asynchronous jobs."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
