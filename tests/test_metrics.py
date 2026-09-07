"""Stage 2 metrics tests: exact metric math plus a cheap offline smoke.

The unit tests hand-build Forecast objects so the metric definitions are
verified exactly. The integration tests reuse walk-forward results already
covered by tests/test_no_leakage.py (persistence on the committed fixture) and
one short in-memory GBM run, keeping added make-test cost small.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from stock_prediction.metrics import (
    compute_metrics,
    compute_result_metrics,
    metrics_by_window,
    window_slices,
)
from stock_prediction.models import GradientBoostedReturnModel, PersistenceModel
from stock_prediction.walkforward import Forecast, WalkForwardResult, run_walkforward


def make_forecast(day: int, last_close: float, pred_ret: float, actual_close: float) -> Forecast:
    return Forecast(
        origin=pd.Timestamp("2024-01-02") + pd.Timedelta(days=day),
        last_close=last_close,
        predicted_return=pred_ret,
        predicted_close=last_close * math.exp(pred_ret),
        actual_close=actual_close,
    )


def test_metric_math_on_hand_built_forecasts():
    # Realized returns: log(102/100) ~= +0.0198026, log(99/102) ~= -0.0298529.
    forecasts = [
        make_forecast(1, 100.0, +0.01, 102.0),  # long, realized up -> directional hit
        make_forecast(2, 102.0, +0.02, 99.0),  # long, realized down -> miss
        make_forecast(3, 99.0, -0.01, 97.0),  # short, realized down -> hit
        make_forecast(4, 97.0, 0.0, 98.0),  # flat call: excluded from directional
    ]
    report = compute_metrics("test", forecasts)

    realized = np.array(
        [math.log(102 / 100), math.log(99 / 102), math.log(97 / 99), math.log(98 / 97)]
    )
    pred = np.array([0.01, 0.02, -0.01, 0.0])
    err = pred - realized
    assert report.rmse == pytest.approx(math.sqrt(np.mean(err**2)))
    assert report.mae == pytest.approx(np.mean(np.abs(err)))
    assert report.n == 4
    assert report.n_zero_predicted == 1
    assert report.n_directional == 3  # flat call excluded; three side-taking forecasts
    assert report.directional_accuracy == pytest.approx(2 / 3)
    # edge = mean of (+r1, +r2, -r3, 0*r4) with signs (+,+, -, 0)
    signs = np.array([1.0, 1.0, -1.0, 0.0])
    assert report.edge == pytest.approx(np.mean(signs * realized))
    assert report.mean_realized_return == pytest.approx(np.mean(realized))
    assert report.first_origin == forecasts[0].origin
    assert report.last_origin == forecasts[-1].origin


def test_all_flat_predictions_have_undefined_directional_accuracy_and_zero_edge():
    # Persistence behavior: predicted return is always exactly zero. It takes
    # no side, so directional accuracy must be None (not 0) and edge must be
    # exactly 0.0 -- long/flat that is never long earns nothing.
    forecasts = [make_forecast(i, 100.0, 0.0, 100.0 + (1 if i % 2 else -1)) for i in range(1, 5)]
    report = compute_metrics("persistence", forecasts)
    assert report.directional_accuracy is None
    assert report.n_zero_predicted == report.n == 4
    assert report.n_directional == 0
    assert report.edge == 0.0


def test_window_slices_partition_forecasts():
    forecasts = [make_forecast(i, 100.0, 0.0, 100.0) for i in range(1, 8)]  # 7 forecasts
    slices = window_slices(forecasts)
    assert len(slices["first_half"]) == 3
    assert len(slices["second_half"]) == 4
    assert slices["full"] == forecasts
    assert slices["first_half"] + slices["second_half"] == forecasts
    assert slices["first_half"][-1].origin < slices["second_half"][0].origin


def test_metrics_by_window_covers_full_and_halves():
    forecasts = [make_forecast(i, 100.0, 0.0, 100.0) for i in range(1, 9)]
    result = WalkForwardResult(model_name="persistence", forecasts=tuple(forecasts))
    reports = metrics_by_window(result)
    assert set(reports) == {"full", "first_half", "second_half"}
    assert reports["full"].n == 8
    assert reports["first_half"].n + reports["second_half"].n == 8


def load_committed_closes():
    from stock_prediction.data import load_csv

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_daily.csv")
    return load_csv(fixture)


def test_metrics_smoke_on_committed_fixture_persistence():
    """Cheap end-to-end smoke: persistence through the harness on the original
    fixture, then metrics. Persistence predicts zero returns, so edge is exactly
    0.0 and RMSE equals the RMS of realized returns."""
    closes = load_committed_closes()
    result = run_walkforward(PersistenceModel(), closes)
    report = compute_result_metrics(result)
    realized = np.array([np.log(f.actual_close / f.last_close) for f in result.forecasts])
    assert report.n == result.n_forecasts > 0
    assert report.edge == 0.0
    assert report.rmse == pytest.approx(math.sqrt(np.mean(realized**2)))
    assert report.mae == pytest.approx(np.mean(np.abs(realized)))
    assert report.directional_accuracy is None
    windows = metrics_by_window(result)
    assert windows["full"].n == windows["first_half"].n + windows["second_half"].n


def test_metrics_smoke_on_short_gbm_run():
    """End-to-end GBM metrics smoke on a short in-memory series: fast enough
    for make test while proving metrics consume real GBM walk-forward output."""
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0002, 0.01, size=110)
    closes = pd.Series(
        100.0 * np.exp(np.cumsum(rets)),
        index=pd.bdate_range("2024-01-02", periods=110),
        name="close",
    )
    result = run_walkforward(GradientBoostedReturnModel(), closes)
    report = compute_result_metrics(result)
    assert report.n > 0
    assert math.isfinite(report.rmse) and math.isfinite(report.mae)
    assert math.isfinite(report.edge)
    assert report.n_zero_predicted < report.n  # GBM takes sides on at least some rows


def test_compute_metrics_rejects_empty():
    with pytest.raises(ValueError, match="at least one forecast"):
        compute_metrics("persistence", [])
