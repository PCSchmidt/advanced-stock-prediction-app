"""Feature-level leak-detection tests.

Guard 1: features for target time t must not change when closes at t or later
change. Mutates the tail of a copy of the fixture and compares feature rows,
exercising `features.build_features` for real. (Harness fit-isolation guards
are added with the walk-forward module.)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from stock_prediction.features import build_features

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_daily.csv")


def load_closes():
    from stock_prediction.data import load_csv

    return load_csv(FIXTURE)


def test_fixture_exists_and_is_small():
    assert os.path.exists(FIXTURE)
    n_rows = len(load_closes())
    assert 50 <= n_rows <= 1000, f"fixture should stay small, has {n_rows} rows"


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
