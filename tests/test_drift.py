"""Stage 5 drift detector tests: deterministic, offline, docker-free.

The detector is pure numpy/scipy on committed fixtures (or small in-memory
series), so every test here is fast. The critical assertions:

- a KNOWN shifted pair fires (vol_regime_shift halves: volatility doubles
  halfway through that fixture by construction);
- matched windows stay quiet (halves of the stationary fixtures, a series
  compared with itself, and sample_daily vs mean_reverting, whose marginal
  return distributions are close);
- the PSI math itself is exact on hand-built arrays.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from stock_prediction.drift import (
    KS_ALPHA,
    MIN_WINDOW_ROWS,
    PSI_RETRAIN_THRESHOLD,
    DriftReport,
    compare_windows,
    detect_drift,
    population_stability_index,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name: str) -> pd.Series:
    from stock_prediction.data import load_csv

    return load_csv(os.path.join(FIXTURE_DIR, f"{name}.csv"))


def random_walk_closes(seed: int, n: int = 300, sigma: float = 0.01) -> pd.Series:
    """Deterministic stationary synthetic walk (matched-window test data)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, size=n)
    return pd.Series(
        100.0 * np.exp(np.cumsum(rets)),
        index=pd.bdate_range("2024-01-02", periods=n),
        name="close",
    )


# --- PSI math (exact, hand-checked) ---------------------------------------


def test_psi_is_zero_for_identical_samples():
    x = np.linspace(-3.0, 3.0, 200)
    assert population_stability_index(x, x) == 0.0


def test_psi_grows_with_shift_and_crosses_threshold():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=500)
    tiny = rng.normal(0.0005, 0.01, size=500)  # 0.05 sigma shift
    big = rng.normal(0.02, 0.01, size=500)  # 2 sigma shift
    psi_zero = population_stability_index(base, rng.normal(0.0, 0.01, size=500))
    psi_tiny = population_stability_index(base, tiny)
    psi_big = population_stability_index(base, big)
    assert psi_tiny > psi_zero >= 0.0
    assert psi_big > psi_tiny
    assert psi_big > PSI_RETRAIN_THRESHOLD
    assert psi_tiny < PSI_RETRAIN_THRESHOLD


def test_psi_rejects_empty_windows():
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(np.array([]), np.array([1.0]))


def test_psi_constant_expected_is_zero():
    assert population_stability_index(np.zeros(100), np.zeros(100)) == 0.0


# --- detector on committed fixtures ----------------------------------------


def test_detector_fires_on_known_shifted_fixture_halves():
    """vol_regime_shift doubles sigma halfway: the detector MUST fire."""
    report = detect_drift(load("vol_regime_shift"), label="vol_regime_shift")
    assert isinstance(report, DriftReport)
    assert report.fired is True
    assert report.retrain_recommended is True
    comp = report.comparisons[0]
    assert comp.feature == "log_return"
    assert comp.psi >= PSI_RETRAIN_THRESHOLD  # recorded: 1.13, significant band
    assert comp.ks_pvalue <= 0.01


def test_detector_stays_quiet_on_matched_stationary_halves():
    """Halves of each stationary fixture are matched windows: MUST NOT fire."""
    for name in ("sample_daily", "trending_up", "mean_reverting"):
        report = detect_drift(load(name), label=name)
        assert report.fired is False, name
        assert report.comparisons[0].psi < PSI_RETRAIN_THRESHOLD, name


def test_detector_stays_quiet_on_synthetic_stationary_halves():
    """A freshly generated stationary walk: matched halves must not fire.

    Seed 7 recorded: PSI 0.026 (stable band), KS p 0.89.
    """
    report = detect_drift(random_walk_closes(seed=7), label="synthetic")
    assert report.fired is False
    assert report.comparisons[0].band == "stable"
    assert report.comparisons[0].ks_pvalue > KS_ALPHA


def test_cross_series_comparison_fires_and_stays_quiet():
    sample = load("sample_daily")
    fired = compare_windows(
        sample,
        load("vol_regime_shift"),
        expected_label="sample_daily",
        actual_label="vol_regime_shift",
    )
    assert fired.fired is True  # KS p ~ 0.004, PSI moderate
    quiet = compare_windows(
        sample,
        load("mean_reverting"),
        expected_label="sample_daily",
        actual_label="mean_reverting",
    )
    assert quiet.fired is False  # close marginals: PSI 0.066, KS p 0.145
    self_cmp = compare_windows(sample, sample, expected_label="a", actual_label="a")
    assert self_cmp.fired is False
    assert self_cmp.comparisons[0].psi == 0.0


def test_mean_drift_pair_records_moderate_psi_and_fires_on_ks():
    """sample_daily vs trending_up: documented pure-mean-shift comparison.

    Recorded behavior: PSI 0.239 (moderate band, below the retrain line) but
    KS p 0.003 <= KS_ALPHA, so the comparison fires. This pins BOTH numbers.
    """
    report = compare_windows(
        load("sample_daily"),
        load("trending_up"),
        expected_label="sample_daily",
        actual_label="trending_up",
    )
    comp = report.comparisons[0]
    assert 0.1 <= comp.psi < PSI_RETRAIN_THRESHOLD
    assert comp.band == "moderate"
    assert comp.fired is True  # via the KS criterion
    assert report.retrain_recommended is True


# --- guards ----------------------------------------------------------------


def test_detect_drift_rejects_short_series():
    with pytest.raises(ValueError, match="at least"):
        detect_drift(random_walk_closes(seed=1, n=2 * MIN_WINDOW_ROWS - 1))


def test_compare_windows_rejects_short_window():
    with pytest.raises(ValueError, match="need at least"):
        compare_windows(
            random_walk_closes(seed=1, n=MIN_WINDOW_ROWS - 1),
            random_walk_closes(seed=2, n=200),
            expected_label="a",
            actual_label="b",
        )


def test_report_as_dict_shape():
    report = detect_drift(load("sample_daily"), label="sample_daily")
    d = report.as_dict()
    assert d["fired"] is False
    assert d["retrain_recommended"] is False
    assert d["thresholds"]["psi_retrain_at_or_above"] == PSI_RETRAIN_THRESHOLD
    assert d["window_definition"].startswith("first half vs second half")
    assert len(d["features"]) == 1
    assert {"feature", "psi", "ks_statistic", "ks_pvalue", "band", "fired"} <= set(d["features"][0])
