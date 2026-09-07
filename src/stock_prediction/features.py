"""Feature engineering with a strict no-lookahead contract.

Row `t` of the feature matrix contains features computed ONLY from closes at
times strictly before `t` (`close[t-1]` and earlier). The label at row `t` is
the close at time `t`. This means: any change to closes at times `>= t` cannot
affect row `t`'s features. `tests/test_no_leakage.py` enforces this contract.

Features (one-step-ahead return forecasting):

- `ret_1`: log return over the previous day (close[t-1] / close[t-2]).
- `lag_{2,3,4,5}`: log returns lagged by 2..5 days.
- `roll_mean_10`: mean of the last 10 daily log returns (ending at t-1).
- `roll_std_10`: std of the last 10 daily log returns (ending at t-1).
- `roll_mean_20`, `roll_std_20`: same over 20 days.

Rationale: daily log returns are approximately stationary (raw prices are not),
lags give the model short-term momentum, and rolling mean/std summarize recent
volatility and trend. All transforms use only past data, so the matrix is safe
for walk-forward fitting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 4, 5)
WINDOWS = (10, 20)

FEATURE_NAMES = (
    [f"ret_lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + [f"roll_std_{w}" for w in WINDOWS]
)


def log_returns(closes: pd.Series) -> pd.Series:
    """Daily log returns: ln(close[t] / close[t-1]). First value is NaN."""
    return np.log(closes.astype(float)).diff()


def build_features(closes: pd.Series) -> pd.DataFrame:
    """Return a feature table indexed by *target time*.

    Row at date `t` holds features from closes strictly before `t`, and is the
    input for predicting `close[t]`. Early rows without enough history are NaN
    and must be dropped by the caller (see `walkforward.make_frames`).
    """
    closes = closes.astype(float)
    rets = log_returns(closes)
    features = pd.DataFrame(index=closes.index)
    for k in LAGS:
        features[f"ret_lag_{k}"] = rets.shift(k)
    for w in WINDOWS:
        features[f"roll_mean_{w}"] = rets.rolling(w).mean().shift(1)
        features[f"roll_std_{w}"] = rets.rolling(w).std().shift(1)
    return features[FEATURE_NAMES]


def build_targets(closes: pd.Series) -> pd.Series:
    """Labels: the close at each row's target time `t` (no transformation)."""
    return closes.astype(float).rename("target_close")


def make_frames(closes: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Assemble (X, y, last_close) with NaN history rows dropped.

    `last_close[t]` is `close[t-1]`: the most recent price known when
    forecasting time `t`. Models use it to turn predicted returns into prices.
    """
    features = build_features(closes)
    targets = build_targets(closes)
    last_close = closes.shift(1).rename("last_close")
    valid = features.dropna().index.intersection(targets.dropna().index)
    valid = valid.intersection(last_close.dropna().index)
    return features.loc[valid], targets.loc[valid], last_close.loc[valid]
