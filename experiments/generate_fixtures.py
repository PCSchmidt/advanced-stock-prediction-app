"""Generate the additional committed synthetic evaluation fixtures (Stage 2).

Creates three ~300-row daily (date, close) CSVs under tests/fixtures/, each a
different deterministic regime, so Stage 2 compares persistence vs
hist_gradient_boosting across multiple offline series without any network
access:

- trending_up.csv        : random walk with constant positive drift (seed 42).
- mean_reverting.csv     : AR(1) log price pulled toward a fixed level (seed 7).
- vol_regime_shift.csv   : zero-drift random walk whose volatility doubles
                           halfway through (seed 11).

These are educational synthetic walks, not market data. Regeneration is
deterministic: run this script with the project venv.

    .venv/Scripts/python experiments/generate_fixtures.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "tests", "fixtures")
N_ROWS = 300
START_CLOSE = 100.0


def _dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=N_ROWS)


def _write(name: str, closes: np.ndarray) -> None:
    path = os.path.join(FIXTURE_DIR, name)
    df = pd.DataFrame({"date": _dates().strftime("%Y-%m-%d"), "close": np.round(closes, 4)})
    df.to_csv(path, index=False)
    print(f"wrote {path} rows={len(df)} first={df['close'].iloc[0]} last={df['close'].iloc[-1]}")


def trending_up() -> None:
    """Constant positive drift: r_t ~ Normal(mu=0.0008, sigma=0.01), seed 42."""
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0008, 0.01, size=N_ROWS)
    _write("trending_up.csv", START_CLOSE * np.exp(np.cumsum(rets)))


def mean_reverting() -> None:
    """AR(1) log price: x_t = m + phi * (x_{t-1} - m) + eps, phi=0.85,
    m=log(100), eps ~ Normal(0, 0.012), seed 7. Daily returns are negatively
    autocorrelated, so a model that learns the pull-back can in principle
    beat persistence here."""
    rng = np.random.default_rng(7)
    eps = rng.normal(0.0, 0.012, size=N_ROWS)
    m = np.log(START_CLOSE)
    x = np.empty(N_ROWS)
    x[0] = m + eps[0]
    for t in range(1, N_ROWS):
        x[t] = m + 0.85 * (x[t - 1] - m) + eps[t]
    _write("mean_reverting.csv", np.exp(x))


def vol_regime_shift() -> None:
    """Zero drift, sigma=0.005 for the first half and sigma=0.02 for the
    second half (a quiet-then-volatile regime), seed 11."""
    rng = np.random.default_rng(11)
    sigma = np.concatenate([np.full(N_ROWS // 2, 0.005), np.full(N_ROWS - N_ROWS // 2, 0.02)])
    rets = rng.normal(0.0, 1.0, size=N_ROWS) * sigma
    _write("vol_regime_shift.csv", START_CLOSE * np.exp(np.cumsum(rets)))


def main() -> None:
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    trending_up()
    mean_reverting()
    vol_regime_shift()


if __name__ == "__main__":
    main()
