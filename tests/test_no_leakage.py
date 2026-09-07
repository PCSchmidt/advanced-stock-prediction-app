"""Leak-detection tests. These fail if a naive leak is introduced.

Two independent guards:

1. Feature invariance: features for target time t must not change when closes
   at t or later change. Mutates the tail of a copy of the fixture and compares
   feature rows, exercising `features.build_features` for real.
2. Harness fit isolation: a spy model records every (fit, predict) call. The
   harness must never pass a training row whose target time is at or after the
   origin into `fit`, and must actually call fit+predict once per origin, so
   the guard cannot pass vacuously.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from stock_prediction.features import build_features, make_frames
from stock_prediction.models import GradientBoostedReturnModel, PersistenceModel
from stock_prediction.walkforward import run_walkforward

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_daily.csv")


def load_closes() -> pd.Series:
    from stock_prediction.data import load_csv

    return load_csv(FIXTURE)


def test_fixture_exists_and_is_small():
    assert os.path.exists(FIXTURE)
    n_rows = len(load_closes())
    assert 50 <= n_rows <= 1000, f"fixture should stay small, has {n_rows} rows"


# --- Guard 1: features at time t are invariant to data at t and later -------


@pytest.mark.parametrize("perturb_start", [100, 200, 280])
def test_features_invariant_to_changes_at_or_after_target_time(perturb_start):
    """Features at target time t must be invariant to ANY change in closes at
    t or later. This is the 'strictly before t' contract: a feature that reads
    close[t] itself (target leakage) changes row t under this perturbation and
    must fail here, not just rows past t+1."""
    closes = load_closes()
    perturbed = closes.copy()
    tail = perturbed.index[perturb_start:]
    perturbed.loc[tail] = perturbed.loc[tail] * 1.37 + 11.0  # arbitrary change

    X_orig = build_features(closes)
    X_pert = build_features(perturbed)

    # Rows targeting times <= perturb_start must be byte-identical: their
    # features may only use closes strictly before their target time.
    through_k = X_orig.index[: perturb_start + 1]
    pd.testing.assert_frame_equal(X_orig.loc[through_k], X_pert.loc[through_k])

    # Rows targeting times strictly AFTER perturb_start legitimately use the
    # changed closes; asserting they differ proves the test has teeth.
    after_k = X_orig.index[perturb_start + 1 :]
    changed = (X_orig.loc[after_k].to_numpy() != X_pert.loc[after_k].to_numpy()).any()
    assert changed, "perturbation unexpectedly changed nothing; test is vacuous"


def test_features_use_only_strictly_prior_closes():
    """ret_lag_1 at target time t must equal the log return of close[t-1]."""
    closes = load_closes()
    X = build_features(closes)
    rets = np.log(closes).diff()
    sample = X.index[80:90]
    expected = rets.shift(1).loc[sample]  # return of day t-1, not day t
    pd.testing.assert_series_equal(X.loc[sample, "ret_lag_1"], expected, check_names=False)


# --- Guard 2: the harness never fits on data at or after the origin ---------


class SpyModel:
    """Records the last target time seen in each fit and every predict origin."""

    name = "spy"

    def __init__(self) -> None:
        self.fit_max_times: list[pd.Timestamp] = []
        self.fit_min_times: list[pd.Timestamp] = []
        self.predict_origins: list[pd.Timestamp] = []
        self.n_fit_calls = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SpyModel:
        self.n_fit_calls += 1
        self.fit_max_times.append(y.index.max())
        self.fit_min_times.append(y.index.min())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert len(X) == 1, "harness must predict exactly the next-step row"
        self.predict_origins.append(X.index[0])
        return np.zeros(len(X))


def test_harness_never_fits_on_data_at_or_after_origin():
    spy = SpyModel()
    result = run_walkforward(spy, load_closes())

    origins = [f.origin for f in result.forecasts]
    assert origins == spy.predict_origins, (
        "harness must predict exactly one row per origin, in order"
    )
    assert spy.n_fit_calls == len(origins) > 0, "harness must fit once per origin (non-vacuous)"

    for i, origin in enumerate(origins):
        assert spy.fit_max_times[i] < origin, (
            f"fit at origin {origin} saw training target {spy.fit_max_times[i]} "
            "(data at or after the origin leaked into fit)"
        )
        assert spy.fit_min_times[i] <= spy.fit_max_times[i]


def test_harness_training_window_is_expanding_and_ordered():
    spy = SpyModel()
    run_walkforward(spy, load_closes())
    fit_max = spy.fit_max_times
    assert fit_max == sorted(fit_max), "origins must roll forward monotonically"
    assert len(set(fit_max)) == len(fit_max), "each origin must have a distinct training cutoff"


def test_forecast_uses_known_last_close_only():
    """predicted_close must be last_close * exp(pred_return), with last_close
    equal to close[origin - 1] -- never the actual close at the origin."""
    closes = load_closes()
    result = run_walkforward(PersistenceModel(), closes)
    _, _, last_close = make_frames(closes)
    for f in result.forecasts:
        assert f.last_close == last_close.loc[f.origin]
        assert f.predicted_close == pytest.approx(f.last_close * np.exp(f.predicted_return))


def test_persistence_forecasts_last_close():
    closes = load_closes()
    result = run_walkforward(PersistenceModel(), closes)
    for f in result.forecasts:
        assert f.predicted_return == 0.0
        assert f.predicted_close == pytest.approx(f.last_close)


def test_primary_model_fits_and_predicts_through_harness():
    closes = load_closes()
    result = run_walkforward(GradientBoostedReturnModel(), closes)
    assert result.n_forecasts > 0
    assert all(np.isfinite(f.predicted_return) for f in result.forecasts)
