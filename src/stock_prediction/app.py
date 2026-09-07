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
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .metrics import MetricReport, compute_result_metrics
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
    version="0.1.0",
)


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
    """POST /forecast response. Small by design: summary stats + last-k only."""

    source: str
    n_rows: int
    models: list[ModelSummary]


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
def forecast(req: ForecastRequest) -> ForecastResponse:
    """Run the real walk-forward harness on a committed fixture (or live data,
    if double-gated on) and return per-model summary stats + the last-k forecasts."""
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

    return ForecastResponse(
        source=source_label,
        n_rows=len(closes),
        models=[_summary(r, compute_result_metrics(r), req.last_k) for r in results],
    )
