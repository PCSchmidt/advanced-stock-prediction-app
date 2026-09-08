"""Structured JSON logging + in-process service metrics (Stage 5).

One tiny observability layer shared by the FastAPI app (``app.py``) and the
CLI (``cli.py``):

- **Logging.** ``configure_logging`` puts a single stdout handler with
  ``JsonFormatter`` on the ``stock_prediction`` logger, so every event leaves
  as ONE json line: timestamp, level, event name, plus structured fields
  (request id, endpoint, latency, error class, drift signal). No secrets
  exist in this app and none are logged; fields are an allowlist, not a dump.
- **Metrics.** ``RequestMetrics`` is an in-process counter store (request
  count, error count, bounded latency sample, last drift signal). It is
  per-process memory -- it resets on restart and aggregates nothing across
  replicas. ``GET /metrics`` in ``app.py`` renders its snapshot as JSON.
  Phase 2 adds ``prom.py``: a stdlib-only Prometheus text writer for the four
  generic HTTP families, served at ``GET /metrics/prometheus`` -- still
  per-process memory, still no Prometheus/Grafana stack behind it.

uvicorn's own access/error logging is untouched; these JSON lines are
additional, one per request handled by this app's middleware.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np

LOGGER_NAME = "stock_prediction"

# Structured fields allowed on a log record (everything else is ignored by the
# formatter). Explicit allowlist: no request bodies, no stack dumps, no
# environment dumps -- there are no secrets in this app and none must leak.
LOG_FIELDS = (
    "event",
    "request_id",
    "method",
    "endpoint",
    "status",
    "latency_ms",
    "error_class",
    "drift_signal",
    "fixture",
    "model",
    "n_forecasts",
    # Phase 3: one startup line reports what the eval-gauge loader loaded.
    "eval_fixture",
    "eval_window",
    "eval_rows_loaded",
    "sklearn_version",
)

_configured = False


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
        }
        if record.exc_info:
            payload["error_class"] = (
                record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
            )
        for field in LOG_FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        message = record.getMessage()
        if message:
            payload["message"] = message
        return json.dumps(payload, default=str)


def configure_logging(*, level: int = logging.INFO) -> logging.Logger:
    """Idempotently attach the stdout JSON handler to the package logger."""
    global _configured
    logger = logging.getLogger(LOGGER_NAME)
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
        _configured = True
    return logger


def log_event(event: str, **fields: object) -> None:
    """Emit one structured JSON event line to stdout."""
    configure_logging().info("", extra={"event": event, **fields})


@contextmanager
def log_context(level: int = logging.INFO) -> Iterator[logging.Logger]:
    """Yield the package logger, configured (CLI convenience)."""
    yield configure_logging(level=level)


class RequestMetrics:
    """Thread-safe in-process request counters and bounded latency sample."""

    def __init__(self, max_latency_samples: int = 1000) -> None:
        self._lock = threading.Lock()
        self._max_latency_samples = max_latency_samples
        self.request_count = 0
        self.error_count = 0
        self._latencies_ms: deque[float] = deque(maxlen=max_latency_samples)
        self.last_drift: dict[str, object] | None = None

    def record(self, *, status: int, latency_ms: float) -> None:
        with self._lock:
            self.request_count += 1
            self._latencies_ms.append(float(latency_ms))
            if status >= 400:
                self.error_count += 1

    def set_drift(self, report: dict[str, object]) -> None:
        with self._lock:
            self.last_drift = report

    def reset(self) -> None:
        """Test isolation hook: return the counters to a fresh state."""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self._latencies_ms.clear()
            self.last_drift = None

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            latencies = np.asarray(self._latencies_ms, dtype=float)
            if latencies.size:
                latency_summary: dict[str, object] = {
                    "count": int(latencies.size),
                    "mean_ms": round(float(np.mean(latencies)), 3),
                    "p50_ms": round(float(np.percentile(latencies, 50)), 3),
                    "p95_ms": round(float(np.percentile(latencies, 95)), 3),
                    "p99_ms": round(float(np.percentile(latencies, 99)), 3),
                }
            else:
                latency_summary = {"count": 0}
            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": (
                    round(self.error_count / self.request_count, 6) if self.request_count else 0.0
                ),
                "latency_ms": latency_summary,
                "last_drift": self.last_drift,
                "note": (
                    "in-process counters since server start; no persistence, "
                    "no cross-replica aggregation, no Prometheus/Grafana"
                ),
            }


def now_ms() -> float:
    """Monotonic wall clock in milliseconds (latency measurement)."""
    return time.perf_counter() * 1000.0
