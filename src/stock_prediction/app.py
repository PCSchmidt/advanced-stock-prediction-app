"""FastAPI serving layer (Stage 4): health check + offline walk-forward forecasts.

A thin HTTP wrapper around the UNCHANGED Stage 1 walk-forward harness
(`walkforward.py`) and Stage 2 metrics (`metrics.py`). The same persistence and
HistGradientBoosting models run through the same expanding-origin loop the CLI
and `make eval` use -- no model changes, no tuning, no second implementation.

Offline by default. Forecasts run on committed (date, close) fixtures from
`tests/fixtures/` (see STOCK_PREDICTION_FIXTURE_DIR below). A live yfinance
path exists but is double-gated: the caller must send
`{"source": "live", "symbol": "..."}` AND the server must be started with
`ALLOW_LIVE_DATA=1`. No API keys are used or baked in anywhere.

Run locally:

    uvicorn stock_prediction.app:app --host 127.0.0.1 --port 8000

Interactive request/response schema docs are served at /docs (Swagger UI).

Stage 5 monitoring (additive; nothing about /health or /forecast changed):

- Every POST /forecast response carries a `drift` object: the Stage 5
  detector (drift.py) run on first-half vs second-half of the closes that
  were just forecast, or an explicit "skipped" note when the series is too
  short for two stable windows. The detector only READS closes; a fired
  signal changes nothing about the forecasts.
- GET /drift runs the same detector on demand: on one fixture's halves by
  default, or between two fixtures with ?baseline=<fixture>.
- GET /metrics returns in-process JSON counters: request count, latency
  p50/p95/p99, error rate, and the last drift signal. Per-process memory
  only; no Prometheus/Grafana, no persistence, no alerting.
- Every request logs ONE structured JSON line to stdout (obs.py): request
  id, method, endpoint, status, latency in ms, error class, and the drift
  signal when a drift path ran. No secrets exist here and none are logged.

Status codes:
- 200: success (health or forecast).
- 400: semantically invalid request the schema cannot express -- unknown
  fixture name, missing `symbol` with `source="live"`, or a harness error
  (e.g. not enough usable rows).
- 403: `source="live"` requested while the server was not started with
  ALLOW_LIVE_DATA=1.
- 422: request body fails schema validation (FastAPI/pydantic; wrong types,
  unknown `model` value, `max_rows` below the walk-forward minimum).
- 500: unexpected server-side failure (e.g. fixture directory misconfigured).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .drift import MIN_WINDOW_ROWS, DriftReport, compare_windows, detect_drift
from .metrics import MetricReport, compute_result_metrics
from .obs import RequestMetrics, log_event, now_ms
from .walkforward import Forecast, run_all_models

# Committed offline fixtures shipped with the repo. Names only -- the API never
# accepts arbitrary filesystem paths.
KNOWN_FIXTURES = ("sample_daily", "trending_up", "mean_reverting", "vol_regime_shift")

app = FastAPI(
    title="stock_prediction API",
    description=(
        "Educational one-step-ahead stock forecasting (walk-forward harness, "
        "offline fixtures by default). Not investment advice; not production."
    ),
    version="0.2.0",
)

# Stage 5 observability state: per-process counters + one JSON log line per
# request. Resets on restart; aggregates nothing across processes.
metrics = RequestMetrics()


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Log one structured line per request and feed the /metrics counters."""
    request_id = uuid.uuid4().hex[:12]
    started = now_ms()
    response = None
    try:
        response = await call_next(request)
        return response
    finally:
        status = response.status_code if response is not None else 500
        error_class = f"http_{status}" if status >= 400 else None
        latency_ms = now_ms() - started
        metrics.record(status=status, latency_ms=latency_ms)
        drift_signal = getattr(request.state, "drift_fired", None)
        log_event(
            "http_request",
            request_id=request_id,
            method=request.method,
            endpoint=request.url.path,
            status=status,
            latency_ms=round(latency_ms, 3),
            error_class=error_class,
            drift_signal=drift_signal,
        )
        if response is not None:
            response.headers["x-request-id"] = request_id


def _fixture_dir() -> Path:
    """Directory holding the committed fixture CSVs.

    Resolution order:
    1. STOCK_PREDICTION_FIXTURE_DIR env var (used by docker-compose, where the
       package is installed into site-packages and the source-relative default
       would not resolve).
    2. Source checkout layout: <repo>/tests/fixtures relative to this file.
    """
    env = os.environ.get("STOCK_PREDICTION_FIXTURE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "tests" / "fixtures"


class ForecastRequest(BaseModel):
    """POST /forecast body. Every field is optional."""

    fixture: str = Field(
        default="sample_daily",
        description="Name of a committed fixture CSV (no paths): " + ", ".join(KNOWN_FIXTURES),
    )
    model: Literal["persistence", "hist_gradient_boosting", "both"] = Field(
        default="both", description="Which model(s) to run through the harness"
    )
    max_rows: int | None = Field(
        default=None,
        ge=90,
        description=(
            "Optionally run on only the LAST N rows of the fixture (bounded "
            "harness run; fewer walk-forward origins, faster). The harness and "
            "models are unchanged; the fixture is just truncated from the front. "
            "Omit for the full fixture."
        ),
    )
    last_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="How many of the most recent forecasts to return per model",
    )
    source: Literal["fixture", "live"] = Field(
        default="fixture",
        description="'fixture' (default, offline) or 'live' yfinance fetch",
    )
    symbol: str | None = Field(
        default=None,
        description="Ticker symbol, required iff source='live' (and the server "
        "must have been started with ALLOW_LIVE_DATA=1)",
    )


class ForecastPoint(BaseModel):
    """One walk-forward forecast (mirrors walkforward.Forecast)."""

    origin: str
    last_close: float
    predicted_return: float
    predicted_close: float
    actual_close: float


class ModelSummary(BaseModel):
    """Per-model harness result: counts, Stage 2 metrics, and the last forecasts."""

    model: str
    n_forecasts: int
    rmse: float
    mae: float
    directional_accuracy: float | None
    edge: float
    first_origin: str
    last_origin: str
    final_forecast: ForecastPoint
    recent_forecasts: list[ForecastPoint]


class ForecastResponse(BaseModel):
    """POST /forecast response. Small by design: summary stats + last-k only.

    `drift` (Stage 5, additive) is the detector output for the closes that
    were just forecast -- first half vs second half -- or an explicit
    "skipped" note when the series is too short for two stable windows.
    """

    source: str
    n_rows: int
    models: list[ModelSummary]
    drift: dict[str, object]


def _drift_for_closes(closes, label: str) -> dict[str, object]:
    """Stage 5 drift check on the closes being served (read-only).

    Returns the detector report, or an explicit skipped note when the series
    is shorter than two minimum windows (e.g. max_rows=90 requests).
    """
    if len(closes) < 2 * MIN_WINDOW_ROWS:
        return {
            "status": "skipped",
            "reason": (
                f"drift needs >= {2 * MIN_WINDOW_ROWS} closes for two "
                f"stable windows, got {len(closes)}"
            ),
        }
    report = detect_drift(closes, label=label)
    return report.as_dict()


def _point(f: Forecast) -> ForecastPoint:
    return ForecastPoint(
        origin=str(f.origin.date()),
        last_close=f.last_close,
        predicted_return=f.predicted_return,
        predicted_close=f.predicted_close,
        actual_close=f.actual_close,
    )


def _summary(result, metrics: MetricReport, last_k: int) -> ModelSummary:
    return ModelSummary(
        model=result.model_name,
        n_forecasts=result.n_forecasts,
        rmse=metrics.rmse,
        mae=metrics.mae,
        directional_accuracy=metrics.directional_accuracy,
        edge=metrics.edge,
        first_origin=str(result.forecasts[0].origin.date()),
        last_origin=str(result.forecasts[-1].origin.date()),
        final_forecast=_point(result.forecasts[-1]),
        recent_forecasts=[_point(f) for f in result.forecasts[-last_k:]],
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: the service is up. No model or data work happens here."""
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest, request: Request) -> ForecastResponse:
    """Run the real walk-forward harness on a committed fixture (or live data,
    if double-gated on) and return per-model summary stats, the last-k
    forecasts, and the Stage 5 drift signal for the same closes."""
    if req.source == "live":
        if os.environ.get("ALLOW_LIVE_DATA") != "1":
            raise HTTPException(
                status_code=403,
                detail="live data is disabled on this server; set ALLOW_LIVE_DATA=1 "
                "to enable it (fixtures remain available offline)",
            )
        if not req.symbol:
            raise HTTPException(status_code=400, detail="symbol is required when source='live'")
        from .data import fetch_prices  # lazy: keeps the offline path network-free

        try:
            closes = fetch_prices(req.symbol)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"live fetch failed: {exc}") from exc
        source_label = f"live:{req.symbol}"
    else:
        if req.fixture not in KNOWN_FIXTURES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown fixture {req.fixture!r}; known fixtures: "
                + ", ".join(KNOWN_FIXTURES),
            )
        path = _fixture_dir() / f"{req.fixture}.csv"
        from .data import load_csv

        try:
            closes = load_csv(path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"fixture file {path} not found on this server; check "
                "STOCK_PREDICTION_FIXTURE_DIR",
            ) from exc
        source_label = f"fixture:{req.fixture}"

    if req.max_rows is not None:
        closes = closes.iloc[-req.max_rows :]

    try:
        results = run_all_models(closes, which=req.model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Stage 5 (additive): read-only drift check on the closes just forecast.
    drift_report = _drift_for_closes(closes, source_label)
    request.state.drift_fired = drift_report.get("fired")
    if isinstance(drift_report.get("fired"), bool):
        metrics.set_drift(drift_report)

    return ForecastResponse(
        source=source_label,
        n_rows=len(closes),
        models=[_summary(r, compute_result_metrics(r), req.last_k) for r in results],
        drift=drift_report,
    )


@app.get("/drift")
def drift_endpoint(
    request: Request,
    fixture: str = "sample_daily",
    baseline: str | None = None,
) -> dict[str, object]:
    """Run the Stage 5 detector on demand.

    Default: first half vs second half of `fixture`. With `baseline=<fixture>`,
    compares the full `fixture` series against the full `baseline` series
    instead (e.g. /drift?fixture=vol_regime_shift&baseline=sample_daily).
    The detector only READS committed fixtures; a fired signal is a
    documented recommendation, and nothing here retrains (Stage 6, not built).
    """
    for name in filter(None, (fixture, baseline)):
        if name not in KNOWN_FIXTURES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown fixture {name!r}; known fixtures: " + ", ".join(KNOWN_FIXTURES),
            )
    from .data import load_csv

    fixture_dir = _fixture_dir()
    try:
        closes = load_csv(fixture_dir / f"{fixture}.csv")
        baseline_closes = load_csv(fixture_dir / f"{baseline}.csv") if baseline else None
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="fixture file not found on this server; check STOCK_PREDICTION_FIXTURE_DIR",
        ) from exc

    if baseline_closes is not None:
        report: DriftReport = compare_windows(
            baseline_closes,
            closes,
            expected_label=f"baseline:{baseline}",
            actual_label=f"candidate:{fixture}",
        )
    else:
        report = detect_drift(closes, label=fixture)
    request.state.drift_fired = report.fired
    metrics.set_drift(report.as_dict())
    return report.as_dict()


@app.get("/metrics")
def metrics_endpoint() -> dict[str, object]:
    """In-process service counters: request count, latency percentiles, error
    rate, and the last drift signal. Per-process memory only -- no
    Prometheus/Grafana, no persistence, no alerting (documented honestly)."""
    return metrics.snapshot()
