"""Offline tests for the Stage 4 FastAPI serving layer (tests/test_app.py).

All calls are in-process via fastapi.testclient.TestClient (httpx): no server
process, no network, no API keys. The forecast tests run the REAL Stage 1
walk-forward harness (no mocks): the persistence case uses the full committed
fixture (fast -- the model predicts zero every step), and the both-models case
bounds the run with max_rows=120 so the HistGradientBoosting refits stay inside
CI time budgets. Model/harness code is exactly what the CLI and `make eval`
use; nothing is tuned or stubbed.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from stock_prediction.app import app, metrics

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_metrics():
    """Isolate the module-level Stage 5 counters between tests."""
    metrics.reset()
    yield
    metrics.reset()


def test_health() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_forecast_persistence_full_fixture() -> None:
    """Real harness, full committed fixture, persistence only (fast)."""
    resp = client.post("/forecast", json={"model": "persistence"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "fixture:sample_daily"
    assert body["n_rows"] == 300
    assert len(body["models"]) == 1
    m = body["models"][0]
    assert m["model"] == "persistence"
    assert m["n_forecasts"] == 219  # full-fixture walk-forward origin count
    assert m["first_origin"] == "2024-04-24"
    assert m["last_origin"] == "2025-02-24"
    # persistence predicts zero return: forecast close == last known close
    final = m["final_forecast"]
    assert final["predicted_return"] == 0.0
    assert final["predicted_close"] == pytest.approx(final["last_close"])
    assert len(m["recent_forecasts"]) == 5
    # Stage 2 metrics ride along from metrics.py (README Results: rmse 0.014733)
    assert m["rmse"] == pytest.approx(0.014733, abs=1e-6)
    assert m["edge"] == 0.0
    assert m["directional_accuracy"] is None  # persistence never takes a side


def test_forecast_both_models_bounded() -> None:
    """Real harness on the last 120 fixture rows: persistence + GBM together."""
    resp = client.post("/forecast", json={"model": "both", "max_rows": 120})
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == 120
    assert [m["model"] for m in body["models"]] == [
        "persistence",
        "hist_gradient_boosting",
    ]
    for m in body["models"]:
        assert m["n_forecasts"] == 39  # 120 rows -> 39 usable origins
        assert len(m["recent_forecasts"]) == 5
        assert m["rmse"] >= 0.0
        assert m["mae"] >= 0.0
        assert m["final_forecast"]["origin"] == m["last_origin"]


def test_forecast_unknown_fixture_is_400() -> None:
    resp = client.post("/forecast", json={"fixture": "not_a_fixture"})
    assert resp.status_code == 400
    assert "unknown fixture" in resp.json()["detail"]


def test_forecast_schema_violation_is_422() -> None:
    # max_rows below the walk-forward warm-up minimum -> schema-level rejection
    resp = client.post("/forecast", json={"model": "persistence", "max_rows": 10})
    assert resp.status_code == 422
    # unknown enum value -> 422 too
    resp = client.post("/forecast", json={"model": "deep_learning"})
    assert resp.status_code == 422


def test_forecast_missing_symbol_with_live_source_is_400(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_LIVE_DATA", "1")
    resp = client.post("/forecast", json={"source": "live"})
    assert resp.status_code == 400
    assert "symbol is required" in resp.json()["detail"]


def test_forecast_response_includes_drift_field() -> None:
    """Stage 5 additive field: drift on the closes just forecast (read-only)."""
    resp = client.post("/forecast", json={"model": "persistence"})
    assert resp.status_code == 200
    drift = resp.json()["drift"]
    # sample_daily halves are matched windows: quiet, with the documented numbers
    assert drift["fired"] is False
    assert drift["retrain_recommended"] is False
    assert drift["expected"] == "fixture:sample_daily:first_half"
    assert drift["actual"] == "fixture:sample_daily:second_half"
    assert drift["thresholds"]["psi_retrain_at_or_above"] == 0.25
    assert drift["features"][0]["psi"] == pytest.approx(0.036196, abs=1e-4)


def test_forecast_drift_skipped_on_short_series() -> None:
    # max_rows=90 passes the schema but leaves halves below MIN_WINDOW_ROWS
    resp = client.post("/forecast", json={"model": "persistence", "max_rows": 90})
    assert resp.status_code == 200
    drift = resp.json()["drift"]
    assert drift["status"] == "skipped"
    assert "120" in drift["reason"]


def test_drift_endpoint_fires_on_vol_shift_and_stays_quiet_by_default() -> None:
    quiet = client.get("/drift")
    assert quiet.status_code == 200
    assert quiet.json()["fired"] is False

    fired = client.get("/drift", params={"fixture": "vol_regime_shift"})
    assert fired.status_code == 200
    body = fired.json()
    assert body["fired"] is True
    assert body["worst_feature"]["feature"] == "log_return"
    assert body["worst_feature"]["psi"] == pytest.approx(1.131507, abs=1e-4)


def test_drift_endpoint_cross_fixture_baseline() -> None:
    resp = client.get("/drift", params={"fixture": "vol_regime_shift", "baseline": "sample_daily"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["expected"] == "baseline:sample_daily"
    assert body["actual"] == "candidate:vol_regime_shift"
    assert body["fired"] is True


def test_drift_endpoint_unknown_fixture_is_400() -> None:
    resp = client.get("/drift", params={"fixture": "not_a_fixture"})
    assert resp.status_code == 400
    assert "unknown fixture" in resp.json()["detail"]


def test_metrics_endpoint_counters() -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/drift").status_code == 200
    assert client.post("/forecast", json={"model": "persistence"}).status_code == 200
    assert client.post("/forecast", json={"fixture": "nope"}).status_code == 400
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    # 4 requests so far; the /metrics call itself is recorded by the
    # middleware AFTER this snapshot, so it is not counted in its own body.
    assert body["request_count"] == 4
    assert body["error_count"] == 1
    assert body["error_rate"] == pytest.approx(0.25)
    assert body["latency_ms"]["count"] == 4
    assert body["latency_ms"]["p95_ms"] >= body["latency_ms"]["p50_ms"] >= 0.0
    # the last drift signal rides along from the /drift call above
    assert body["last_drift"]["fired"] is False


def test_structured_json_log_lines() -> None:
    """One structured line per request: request id, endpoint, latency, drift.

    Attaches a capture handler directly to the package logger (it does not
    propagate to root by design), then parses the JSON lines back.
    """
    import io
    import json
    import logging

    from stock_prediction.obs import LOGGER_NAME, JsonFormatter

    logger = logging.getLogger(LOGGER_NAME)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    try:
        client.get("/health")
        client.get("/drift", params={"fixture": "vol_regime_shift"})
    finally:
        logger.removeHandler(handler)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(lines) == 2
    health_line, drift_line = lines
    assert health_line["event"] == "http_request"
    assert health_line["endpoint"] == "/health"
    assert health_line["status"] == 200
    assert "request_id" in health_line and "latency_ms" in health_line
    assert health_line["error_class"] is None
    assert drift_line["endpoint"] == "/drift"
    assert drift_line["drift_signal"] is True  # vol shift fired


def test_forecast_live_source_denied_without_env_gate(monkeypatch) -> None:
    """Live data is refused unless the server was started with ALLOW_LIVE_DATA=1."""
    monkeypatch.delenv("ALLOW_LIVE_DATA", raising=False)
    resp = client.post("/forecast", json={"source": "live", "symbol": "AAPL"})
    assert resp.status_code == 403
    assert "ALLOW_LIVE_DATA" in resp.json()["detail"]
