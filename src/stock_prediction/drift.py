"""Drift detection (Stage 5): PSI + KS on the daily log-return distribution.

Detects distribution shift between two windows of (date, close) data by
comparing the DAILY LOG RETURN distribution -- the quantity the Stage 1
features are built from and the label the models fit. This module only READS
closes; it fits nothing, predicts nothing, changes no model, and implementing
a retrain is explicitly Stage 6 work (NOT done here).

The statistics
--------------
``population_stability_index`` bins the EXPECTED window's returns into
``PSI_BINS`` quantile bins and computes

    PSI = sum_i (a_i - e_i) * ln(a_i / e_i)

where e_i / a_i are the expected/actual sample proportions in bin i (clipped
at ``EPS``). PSI is 0 for identical distributions and grows as the actual
proportions move away from the expected ones. Alongside PSI, a two-sample
Kolmogorov-Smirnov test (``scipy.stats.ks_2samp``) is reported as a
sample-size-aware cross-check.

Why only log returns: the lag features are the same return series shifted by
a constant, and rolling mean/std features overlap so heavily (a 20-day window
shares 19 of its 20 observations with the next one) that a KS test on them is
strongly anti-conservative at this sample size -- first vs second half of the
stationary ``sample_daily`` fixture scores KS p ~ 1e-24 on ``roll_std_20``
from a modest 0.0129 -> 0.0151 std ramp that is plausibly sampling noise.
The log-return series has no such overlap, so its KS p-values mean what they
say. This is a documented design choice, not an oversight.

Window definition
-----------------
``detect_drift(closes)`` splits ONE series into FIRST HALF vs SECOND HALF at
the midpoint row and compares the two halves' log returns.
``compare_windows(expected_closes, actual_closes)`` instead compares two whole
series (e.g. a candidate fixture against a baseline fixture). Each window
needs at least ``MIN_WINDOW_ROWS`` closes.

Thresholds (the documented signal that would trigger a retrain)
---------------------------------------------------------------
The return comparison fires when EITHER holds:

- ``PSI >= PSI_RETRAIN_THRESHOLD`` (0.25), or
- ``KS p-value <= KS_ALPHA`` (0.01).

The 0.25 PSI line is the standard credit-scoring convention (Siddiqi 2006,
*Credit Risk Scorecards*; widely used in model-risk monitoring):
PSI < 0.1 = stable, 0.1-0.25 = moderate shift worth watching, > 0.25 =
significant shift. RETRAIN is recommended only in the significant band or on
strong KS evidence -- a moderate PSI means "investigate", not "rebuild".
Bin-count choice: with ~150 observations per window a 10-bin PSI is
noise-dominated (matched halves of stationary fixtures score PSI ~ 0.32 with
10 bins, above the retrain line), so this detector uses ``PSI_BINS = 5``
quantile bins; matched stationary halves then score PSI < 0.05 while the
known-shifted comparisons still clear the 0.25 / KS lines comfortably.

This module still does not retrain; the drift-gated retrain and rollback
live in maintain.py (Stage 6), executed only on demand.

Recorded fixture behavior (see experiments/drift_log.md and README Monitor):
fires on vol_regime_shift halves (PSI 1.13), on sample_daily vs
vol_regime_shift (KS p 0.004), and on sample_daily vs trending_up mean drift
(KS p 0.003, PSI 0.239 moderate); stays quiet on matched halves of all three
stationary fixtures, on sample_daily vs mean_reverting, and on a series
compared with itself.

Scope honesty: every number this module can produce in this repository comes
from committed SYNTHETIC fixtures, not production traffic. There is no
alerting and no scheduler anywhere in this repository; the only consumer of
the fired signal is the on-demand maintain CLI (maintain.py, Stage 6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

PSI_BINS = 5
PSI_STABLE = 0.1
PSI_RETRAIN_THRESHOLD = 0.25
KS_ALPHA = 0.01
EPS = 1e-6
MIN_WINDOW_ROWS = 60

MONITORED_FEATURES = ("log_return",)

Band = Literal["stable", "moderate", "significant"]


def population_stability_index(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    *,
    n_bins: int = PSI_BINS,
    eps: float = EPS,
) -> float:
    """PSI of ``actual`` against ``expected`` over ``n_bins`` quantile bins.

    Bin edges come from ``expected``'s quantiles; values outside the expected
    range are clipped into the edge bins. Returns 0.0 for degenerate (constant)
    expected data, where binning carries no information.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        msg = "population_stability_index needs non-empty windows"
        raise ValueError(msg)
    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    e_counts = np.histogram(np.clip(expected, edges[0], edges[-1]), edges)[0]
    a_counts = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0]
    e_prop = np.maximum(e_counts / expected.size, eps)
    a_prop = np.maximum(a_counts / actual.size, eps)
    return float(np.sum((a_prop - e_prop) * np.log(a_prop / e_prop)))


def _band(psi: float) -> Band:
    if psi >= PSI_RETRAIN_THRESHOLD:
        return "significant"
    if psi >= PSI_STABLE:
        return "moderate"
    return "stable"


@dataclass(frozen=True)
class FeatureDrift:
    """PSI + KS comparison of ONE monitored distribution across two windows."""

    feature: str
    n_expected: int
    n_actual: int
    psi: float
    ks_statistic: float
    ks_pvalue: float
    band: Band
    fired: bool  # psi >= PSI_RETRAIN_THRESHOLD or ks_pvalue <= KS_ALPHA

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "n_expected": self.n_expected,
            "n_actual": self.n_actual,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "band": self.band,
            "fired": self.fired,
        }


@dataclass(frozen=True)
class DriftReport:
    """Full detector output for one window comparison."""

    expected_label: str
    actual_label: str
    window_definition: str
    comparisons: tuple[FeatureDrift, ...]

    @property
    def fired(self) -> bool:
        return any(c.fired for c in self.comparisons)

    @property
    def retrain_recommended(self) -> bool:
        """True iff the monitored distribution crossed a firing threshold.

        This is a SIGNAL only. The response to it -- drift-gated retrain +
        rollback -- lives in maintain.py (Stage 6) and runs only on demand.
        """
        return self.fired

    @property
    def worst(self) -> FeatureDrift:
        return max(self.comparisons, key=lambda c: c.psi)

    def as_dict(self) -> dict[str, object]:
        return {
            "expected": self.expected_label,
            "actual": self.actual_label,
            "window_definition": self.window_definition,
            "thresholds": {
                "psi_bins": PSI_BINS,
                "psi_stable_below": PSI_STABLE,
                "psi_retrain_at_or_above": PSI_RETRAIN_THRESHOLD,
                "ks_alpha": KS_ALPHA,
                "rule": (
                    "fires when PSI >= 0.25 (significant band) or KS p-value "
                    "<= 0.01; a fired comparison recommends retrain"
                ),
            },
            "fired": self.fired,
            "retrain_recommended": self.retrain_recommended,
            "worst_feature": self.worst.as_dict(),
            "features": [c.as_dict() for c in self.comparisons],
        }


def _log_returns(closes: pd.Series, label: str) -> np.ndarray:
    closes = closes.astype(float)
    if len(closes) < MIN_WINDOW_ROWS:
        msg = (
            f"window {label!r} has {len(closes)} closes, need at least "
            f"{MIN_WINDOW_ROWS} for stable two-window drift statistics"
        )
        raise ValueError(msg)
    return np.log(closes).diff().dropna().to_numpy()


def _compare_returns(expected: np.ndarray, actual: np.ndarray) -> FeatureDrift:
    psi = population_stability_index(expected, actual)
    ks = stats.ks_2samp(expected, actual)
    return FeatureDrift(
        feature="log_return",
        n_expected=int(expected.size),
        n_actual=int(actual.size),
        psi=psi,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        band=_band(psi),
        fired=psi >= PSI_RETRAIN_THRESHOLD or float(ks.pvalue) <= KS_ALPHA,
    )


HALF_SPLIT_DEFINITION = (
    "first half vs second half of one series, split at the midpoint row; "
    "daily log returns computed within each half"
)
CROSS_SERIES_DEFINITION = (
    "full expected series vs full actual series; daily log returns computed within each series"
)


def compare_windows(
    expected_closes: pd.Series,
    actual_closes: pd.Series,
    *,
    expected_label: str,
    actual_label: str,
    window_definition: str = CROSS_SERIES_DEFINITION,
) -> DriftReport:
    """Compare two whole (date, close) series on the log-return distribution."""
    expected = _log_returns(expected_closes, expected_label)
    actual = _log_returns(actual_closes, actual_label)
    return DriftReport(
        expected_label=expected_label,
        actual_label=actual_label,
        window_definition=window_definition,
        comparisons=(_compare_returns(expected, actual),),
    )


def detect_drift(closes: pd.Series, *, label: str = "series") -> DriftReport:
    """Drift check for ONE series: first half vs second half, split at the midpoint."""
    if len(closes) < 2 * MIN_WINDOW_ROWS:
        msg = (
            f"series {label!r} has {len(closes)} closes; drift detection needs at "
            f"least {2 * MIN_WINDOW_ROWS} so each half has {MIN_WINDOW_ROWS}"
        )
        raise ValueError(msg)
    half = len(closes) // 2
    return compare_windows(
        closes.iloc[:half],
        closes.iloc[half:],
        expected_label=f"{label}:first_half",
        actual_label=f"{label}:second_half",
        window_definition=HALF_SPLIT_DEFINITION,
    )
