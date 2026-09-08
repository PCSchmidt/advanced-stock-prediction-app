"""Prometheus text exposition for the generic HTTP serving metrics (Phase 2).

A tiny, dependency-free (stdlib-only) Prometheus text-format writer for the
four generic serving families requested of every portfolio app:

- ``<prefix>requests_total`` counter, labels: endpoint, method, status
- ``<prefix>errors_total`` counter, labels: endpoint, method, error_class
- ``<prefix>request_latency_seconds`` histogram, labels: endpoint, method
- ``<prefix>up`` gauge (1 while the app is serving)

Design rules (deliberate):

- **Low cardinality only.** ``endpoint`` is the ROUTE TEMPLATE (``/forecast``),
  never the full URL; ``status`` is the HTTP status code; ``error_class`` is
  the app's existing bounded vocabulary (``http_400`` ...). No request ids, no
  user symbols, no exception text, no unbounded user content is ever used as a
  label or metric name. Unmatched routes (404s) collapse to ``unmatched``.
- **Deterministic output.** Families and label sets are rendered in sorted
  order, so two calls with the same data produce byte-identical text. This is
  unit-testable without a server.
- **No new dependencies.** The exposition text is hand-rolled per the
  Prometheus text format (media type
  ``text/plain; version=0.0.4; charset=utf-8``). No prometheus_client, no
  lockfile change.
- **Per-process memory only**, like the JSON ``RequestMetrics``: resets on
  restart, aggregates nothing across replicas. No scrape persistence, no
  Grafana, no alerting. ``up`` is always 1 when this module can serve the
  endpoint at all (the only honest value this process can claim).

This phase intentionally adds NO app-domain metrics (drift state, model
quality, etc.) -- only the generic HTTP serving families above.
"""

from __future__ import annotations

import threading

# Prometheus exposition media type for the /metrics/prometheus response.
PROM_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# Default metric prefix for this app.
METRIC_PREFIX = "stock_prediction"

# Documented histogram buckets (seconds) for request latency. The app serves
# fixture-backed forecasts (fast persistence, bounded GBM runs), so the bulk
# of requests land well under a second; the spread covers a 5 ms health check
# up to a 5 s worst case. These are the ONLY latency buckets, fixed at
# import time so the exposition is deterministic.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
)

# Label for requests that matched no route (404s and stray methods). Keeps
# the endpoint label bounded: user-controlled URLs never become label values.
UNMATCHED_ENDPOINT = "unmatched"


def escape_label(value: str) -> str:
    """Escape a label value per the Prometheus text format."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: float) -> str:
    """Deterministic float rendering for the exposition text (_sum)."""
    return f"{value:.6f}"


def format_le(bound: float) -> str:
    """Deterministic compact rendering for histogram `le` bounds (0.005, +Inf)."""
    return repr(float(bound))


def format_labels(labels: dict[str, str]) -> str:
    """Render one sorted label set: {a="1",b="2"}. Empty -> ''."""
    if not labels:
        return ""
    inner = ",".join(f'{k}="{escape_label(v)}"' for k, v in sorted(labels.items()))
    return "{" + inner + "}"


class PrometheusMetrics:
    """Thread-safe store + deterministic renderer for the four families.

    One instance per process (created in ``app.py`` next to the JSON
    ``RequestMetrics``). ``record`` is called by the request middleware;
    ``render`` produces the full exposition text for GET /metrics/prometheus.
    """

    def __init__(
        self,
        prefix: str = METRIC_PREFIX,
        buckets: tuple[float, ...] = LATENCY_BUCKETS,
    ) -> None:
        self._prefix = prefix
        self._buckets = tuple(sorted(buckets))
        self._lock = threading.Lock()
        # (endpoint, method, status) -> count
        self._requests: dict[tuple[str, str, str], int] = {}
        # (endpoint, method, error_class) -> count
        self._errors: dict[tuple[str, str, str], int] = {}
        # (endpoint, method) -> {"counts": [per-bucket <=-bound counts], "sum", "n"}
        self._latency: dict[tuple[str, str], dict[str, object]] = {}

    # ---- ingestion (called by the middleware) --------------------------

    def record(
        self,
        *,
        endpoint: str,
        method: str,
        status: int,
        latency_s: float,
        error_class: str | None = None,
    ) -> None:
        """Record one handled request into all applicable families."""
        ep = endpoint or UNMATCHED_ENDPOINT
        m = method.upper()
        with self._lock:
            req_key = (ep, m, str(status))
            self._requests[req_key] = self._requests.get(req_key, 0) + 1
            if error_class is not None:
                err_key = (ep, m, error_class)
                self._errors[err_key] = self._errors.get(err_key, 0) + 1
            hist = self._latency.setdefault(
                (ep, m),
                {"counts": [0] * len(self._buckets), "sum": 0.0, "n": 0},
            )
            lat = max(0.0, float(latency_s))
            for i, bound in enumerate(self._buckets):
                if lat <= bound:
                    hist["counts"][i] += 1  # type: ignore[index]
            hist["sum"] = float(hist["sum"]) + lat  # type: ignore[arg-type]
            hist["n"] = int(hist["n"]) + 1  # type: ignore[call-overload]

    # ---- rendering -----------------------------------------------------

    def _series(self, name: str, labels: dict[str, str], value: object) -> str:
        return f"{name}{format_labels(labels)} {value}"

    def render(self) -> str:
        """Full Prometheus text exposition. Deterministic (sorted) output."""
        p = self._prefix
        lines: list[str] = []

        lines.append(f"# HELP {p}_requests_total Total HTTP requests handled.")
        lines.append(f"# TYPE {p}_requests_total counter")
        with self._lock:
            requests = sorted(self._requests.items())
            errors = sorted(self._errors.items())
            latency = sorted(self._latency.items())
        for (ep, method, status), count in requests:
            lines.append(
                self._series(
                    f"{p}_requests_total",
                    {"endpoint": ep, "method": method, "status": status},
                    count,
                )
            )

        lines.append(f"# HELP {p}_errors_total Total HTTP requests that ended in an error class.")
        lines.append(f"# TYPE {p}_errors_total counter")
        for (ep, method, err), count in errors:
            lines.append(
                self._series(
                    f"{p}_errors_total",
                    {"endpoint": ep, "method": method, "error_class": err},
                    count,
                )
            )

        lines.append(
            f"# HELP {p}_request_latency_seconds HTTP request latency in seconds "
            f"(buckets in seconds: {', '.join(format_le(b) for b in self._buckets)})."
        )
        lines.append(f"# TYPE {p}_request_latency_seconds histogram")
        for (ep, method), hist in latency:
            base = f"{p}_request_latency_seconds"
            labels = {"endpoint": ep, "method": method}
            for bound, bucket_count in zip(self._buckets, hist["counts"]):
                lines.append(
                    self._series(
                        f"{base}_bucket",
                        {**labels, "le": format_le(bound)},
                        int(bucket_count),
                    )
                )
            lines.append(self._series(f"{base}_bucket", {**labels, "le": "+Inf"}, hist["n"]))
            lines.append(self._series(f"{base}_sum", labels, _fmt(float(hist["sum"]))))
            lines.append(self._series(f"{base}_count", labels, hist["n"]))

        lines.append(f"# HELP {p}_up Whether the app is serving (1 = up).")
        lines.append(f"# TYPE {p}_up gauge")
        lines.append(f"{p}_up 1")
        return "\n".join(lines) + "\n"

    # ---- test isolation --------------------------------------------------

    def reset(self) -> None:
        """Test isolation hook: drop every recorded series."""
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latency.clear()
