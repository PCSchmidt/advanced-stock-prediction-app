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

from stock_prediction.app import app

client = TestClient(app)


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


def test_forecast_live_source_denied_without_env_gate(monkeypatch) -> None:
    """Live data is refused unless the server was started with ALLOW_LIVE_DATA=1."""
    monkeypatch.delenv("ALLOW_LIVE_DATA", raising=False)
    resp = client.post("/forecast", json={"source": "live", "symbol": "AAPL"})
    assert resp.status_code == 403
    assert "ALLOW_LIVE_DATA" in resp.json()["detail"]
