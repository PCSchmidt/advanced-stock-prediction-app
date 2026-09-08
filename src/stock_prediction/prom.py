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

Phase 3 adds the app-domain families on top of the same four, same writer,
same rules:

- ``<prefix>_forecast_requests_total`` counter, labels: ``model`` (the API's
  bounded model vocabulary), ``status`` (HTTP status code). Deliberately a
  separate family rather than a ``model`` label on ``requests_total``: the
  generic HTTP families stay app-agnostic, and ``model`` is only known inside
  the forecast handler, so 422 schema rejections (which never reach the
  handler) are honestly absent here instead of mislabeled.
- ``<prefix>_forecast_latency_seconds`` histogram, labels: ``model``.
- ``<prefix>_drift_checks_total`` counter, label ``outcome`` in
  {fired, quiet}; ``<prefix>_drift_state`` gauge 0/1 = last completed check
  quiet/fired, mirroring the JSON ``last_drift`` semantics. Skipped checks
  are not checks and are not counted.
- ``<prefix>_eval_rmse`` / ``_mae`` / ``_directional_accuracy`` / ``_edge``
  gauges, labels: ``model``. EVALUATION-CONTEXT metrics loaded once at
  startup from the committed Stage 2 artifacts (``experiments/results.csv``);
  the exact evaluation context (fixture, window, row counts) is documented in
  HELP text, NOT as labels, so incompatible evaluation windows can never be
  confusable as label variants.
- ``<prefix>_model_info`` gauge=1, labels: ``model``, ``sklearn_version``.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

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

# ---- Phase 3 bounded vocabularies (labels come ONLY from these) -----------

# The API's `model` request field (see ForecastRequest in app.py). Nothing
# else ever becomes a `model` label: an unexpected value is clamped to
# INVALID_MODEL so user content can never create a label series.
FORECAST_MODELS = ("persistence", "hist_gradient_boosting", "both")
INVALID_MODEL = "invalid"

# Bounded drift-check outcomes, mirroring the JSON last_drift semantics.
DRIFT_OUTCOMES = ("fired", "quiet")

# Fixed evaluation context for the stock_prediction_eval_* gauges. Exactly
# ONE fixture/window combination is exposed; the loader ignores every other
# row of the results CSV so incompatible evaluation windows can never be
# exposed (and therefore never confused). The context is documented in HELP
# text, not as labels.
EVAL_FIXTURE = "sample_daily"
EVAL_WINDOW = "full"
EVAL_MODELS = ("persistence", "hist_gradient_boosting")


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
        # Phase 3 app-domain families. Static (startup-loaded) series live in
        # _eval / _model_info; request-driven series in the dicts below.
        # (model, status) -> count
        self._forecast_requests: dict[tuple[str, str], int] = {}
        # model -> {"counts": [per-bucket <=-bound counts], "sum", "n"}
        self._forecast_latency: dict[str, dict[str, object]] = {}
        # outcome -> count
        self._drift_checks: dict[str, int] = {}
        # 0 = last completed check quiet, 1 = fired; None = no check yet.
        self._drift_state: int | None = None
        # (metric, model) -> value, loaded from the committed Stage 2 CSV.
        self._eval: dict[tuple[str, str], float] = {}
        # Walk-forward origins of the loaded evaluation rows (for HELP text).
        self._eval_n: int = 0
        # (model, sklearn_version) -> 1
        self._model_info: dict[tuple[str, str], int] = {}

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

    def record_forecast(self, *, model: str, status: int, latency_s: float | None = None) -> None:
        """Record one request that reached the forecast handler (any status).

        ``model`` is clamped to the bounded vocabulary (unknown -> "invalid"),
        so user content can never create a label series. Pass ``latency_s``
        only for completed handler runs; early-exit validation errors carry
        no meaningful forecast latency and record none.
        """
        m = model if model in FORECAST_MODELS else INVALID_MODEL
        s = str(status)
        with self._lock:
            key = (m, s)
            self._forecast_requests[key] = self._forecast_requests.get(key, 0) + 1
            if latency_s is not None:
                hist = self._forecast_latency.setdefault(
                    m,
                    {"counts": [0] * len(self._buckets), "sum": 0.0, "n": 0},
                )
                lat = max(0.0, float(latency_s))
                for i, bound in enumerate(self._buckets):
                    if lat <= bound:
                        hist["counts"][i] += 1  # type: ignore[index]
                hist["sum"] = float(hist["sum"]) + lat  # type: ignore[arg-type]
                hist["n"] = int(hist["n"]) + 1  # type: ignore[call-overload]

    def record_drift(self, *, fired: bool) -> None:
        """Record one COMPLETED drift check and mirror it into the state gauge.

        Mirrors the JSON last_drift semantics: skipped checks are never
        recorded (they change neither the counter nor the gauge), and a
        quiet check after a fired one moves the gauge back to 0.
        """
        outcome = "fired" if fired else "quiet"
        with self._lock:
            self._drift_checks[outcome] = self._drift_checks.get(outcome, 0) + 1
            self._drift_state = 1 if fired else 0

    def set_model_info(self, *, model: str, sklearn_version: str) -> None:
        """Register one (model, sklearn_version) identity series (value 1)."""
        with self._lock:
            self._model_info[(model, sklearn_version)] = 1

    def load_eval_gauges_from_csv(self, path: str | Path) -> int:
        """Load the EVALUATION-CONTEXT gauges from the committed Stage 2 CSV.

        Loads ONLY the rows matching the documented context (EVAL_FIXTURE +
        EVAL_WINDOW); every other window in the same file is ignored, so
        incompatible evaluation windows can never be exposed. Idempotent
        (values overwrite). A missing file is not an error: the gauges simply
        stay absent (documented), and startup never fails on a missing
        artifact. Returns the number of rows loaded.
        """
        results_path = Path(path)
        if not results_path.is_file():
            return 0
        loaded = 0
        with results_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("fixture") != EVAL_FIXTURE or row.get("window") != EVAL_WINDOW:
                    continue
                model = row.get("model") or ""
                if model not in EVAL_MODELS:
                    continue
                try:
                    n = int(row["n"])
                    values = {
                        name: float(row[name])
                        for name in ("rmse", "mae", "edge")
                        if row.get(name) not in (None, "")
                    }
                    if row.get("directional_accuracy") not in (None, ""):
                        values["directional_accuracy"] = float(row["directional_accuracy"])
                except (KeyError, TypeError, ValueError):
                    continue
                with self._lock:
                    for name, value in values.items():
                        self._eval[(name, model)] = value
                    self._eval_n = n
                loaded += 1
        return loaded

    def _eval_context(self) -> str:
        """The exact evaluation context, for HELP text (NOT labels)."""
        return (
            f"fixture={EVAL_FIXTURE}, window={EVAL_WINDOW}, n={self._eval_n} walk-forward "
            "origins per model, one-step log returns, expanding-origin harness; source: "
            "committed experiments/results.csv (regenerate with 'make eval'). "
            "EVALUATION-CONTEXT metric from offline artifacts, NOT a live serving metric; "
            "do not compare against any other fixture or window."
        )

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

        # ---- Phase 3 app-domain families (rendered after the generic four) --

        lines.append(
            f"# HELP {p}_forecast_requests_total Total requests that reached the forecast "
            "handler, by bounded model and HTTP status. `model` is the API's bounded "
            f"vocabulary ({', '.join(FORECAST_MODELS)}; unexpected values clamp to "
            f"{INVALID_MODEL}); 422 schema rejections never reach the handler and are "
            "not counted here (the generic requests_total family covers them)."
        )
        lines.append(f"# TYPE {p}_forecast_requests_total counter")
        for (model, status), count in sorted(self._forecast_requests.items()):
            lines.append(
                self._series(
                    f"{p}_forecast_requests_total",
                    {"model": model, "status": status},
                    count,
                )
            )

        lines.append(
            f"# HELP {p}_forecast_latency_seconds End-to-end forecast handler latency in "
            "seconds per requested model (buckets in seconds: "
            f"{', '.join(format_le(b) for b in self._buckets)}). The `model` label is the "
            "REQUESTED model: 'both' runs persistence and HistGradientBoosting in one "
            "handler pass, so its samples are not per-model latencies."
        )
        lines.append(f"# TYPE {p}_forecast_latency_seconds histogram")
        for model, hist in sorted(self._forecast_latency.items()):
            base = f"{p}_forecast_latency_seconds"
            for bound, bucket_count in zip(self._buckets, hist["counts"]):
                lines.append(
                    self._series(
                        f"{base}_bucket",
                        {"le": format_le(bound), "model": model},
                        int(bucket_count),
                    )
                )
            lines.append(self._series(f"{base}_bucket", {"le": "+Inf", "model": model}, hist["n"]))
            lines.append(self._series(f"{base}_sum", {"model": model}, _fmt(float(hist["sum"]))))
            lines.append(self._series(f"{base}_count", {"model": model}, hist["n"]))

        lines.append(
            f"# HELP {p}_drift_checks_total Drift checks actually run by this process. "
            "outcome=fired when the Stage 5 detector fired (PSI >= 0.25 or KS p <= 0.01), "
            "outcome=quiet otherwise. Skipped checks (series too short for two stable "
            "windows) are not checks and are not counted."
        )
        lines.append(f"# TYPE {p}_drift_checks_total counter")
        for outcome, count in sorted(self._drift_checks.items()):
            lines.append(self._series(f"{p}_drift_checks_total", {"outcome": outcome}, count))

        lines.append(
            f"# HELP {p}_drift_state Last COMPLETED drift check: 0 = quiet, 1 = fired "
            "(mirrors the last_drift object of the JSON GET /metrics endpoint). Absent "
            "until the first completed check."
        )
        lines.append(f"# TYPE {p}_drift_state gauge")
        if self._drift_state is not None:
            lines.append(f"{p}_drift_state {self._drift_state}")

        eval_metric_help = {
            "rmse": "Root mean squared error of the predicted next-step log return.",
            "mae": "Mean absolute error of the predicted next-step log return.",
            "directional_accuracy": (
                "Fraction of side-taking forecasts with matching sign of the realized "
                "return. Undefined for persistence (it predicts zero every step; empty "
                "cell in results.csv), so the series is absent for that model."
            ),
            "edge": (
                "Mean(sign(predicted_return) * realized_return) of the long/flat "
                "strategy per step, log-return units, no costs."
            ),
        }
        for metric in sorted({name for name, _ in self._eval}):
            lines.append(
                f"# HELP {p}_eval_{metric} Stage 2 evaluation {metric}: "
                f"{eval_metric_help[metric]} Evaluation context: {self._eval_context()}"
            )
            lines.append(f"# TYPE {p}_eval_{metric} gauge")
            for (name, model), value in sorted(self._eval.items()):
                if name == metric:
                    lines.append(self._series(f"{p}_eval_{metric}", {"model": model}, _fmt(value)))

        lines.append(
            f"# HELP {p}_model_info Model identity metadata, always 1: the bounded model "
            "names and the sklearn version this serving process runs."
        )
        lines.append(f"# TYPE {p}_model_info gauge")
        for (model, version), value in sorted(self._model_info.items()):
            lines.append(
                self._series(
                    f"{p}_model_info",
                    {"model": model, "sklearn_version": version},
                    value,
                )
            )
        return "\n".join(lines) + "\n"

    # ---- test isolation --------------------------------------------------

    def reset(self) -> None:
        """Test isolation hook: drop every recorded series (including the
        static eval/model_info series; tests reload them explicitly)."""
        with self._lock:
            self._requests.clear()
            self._errors.clear()
            self._latency.clear()
            self._forecast_requests.clear()
            self._forecast_latency.clear()
            self._drift_checks.clear()
            self._drift_state = None
            self._eval.clear()
            self._eval_n = 0
            self._model_info.clear()
