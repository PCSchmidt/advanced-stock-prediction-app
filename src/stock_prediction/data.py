"""Data layer: load committed fixtures offline, or fetch live prices with caching.

Design notes:

- `load_csv` is the offline path used by tests and the CLI. It reads a small
  committed CSV (date, close) and returns a pandas Series indexed by date.
- `fetch_prices` is the live path. It imports yfinance lazily inside the
  function so that importing this package never pulls in yfinance or touches
  the network. Fetched prices are cached under `data/cache/` (gitignored) so
  repeat runs work offline for the cache lifetime.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pandas as pd

DEFAULT_CACHE_DIR = os.path.join("data", "cache")
CACHE_TTL = timedelta(hours=24)


def load_csv(path: str | os.PathLike[str]) -> pd.Series:
    """Load a committed (date, close) CSV fixture as a close-price Series.

    Rows are sorted by date and NaN closes are dropped. Raises if the file has
    no usable rows so callers fail loudly instead of forecasting on nothing.
    """
    df = pd.read_csv(path)
    for col in ("date", "close"):
        if col not in df.columns:
            msg = f"fixture CSV must have 'date' and 'close' columns, got {list(df.columns)}"
            raise ValueError(msg)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values("date")
    if df.empty:
        msg = f"fixture CSV {path!r} has no usable rows"
        raise ValueError(msg)
    series = pd.Series(
        df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name="close"
    )
    return series


def _cache_path(cache_dir: str | os.PathLike[str], symbol: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in symbol)
    return os.path.join(str(cache_dir), f"{safe}.csv")


def fetch_prices(
    symbol: str,
    *,
    cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR,
    period: str = "5y",
    ttl: timedelta = CACHE_TTL,
) -> pd.Series:
    """Fetch daily close prices for `symbol` via yfinance, with a file cache.

    The cache is a gitignored CSV under `cache_dir`. If a cache file exists and
    is younger than `ttl`, it is returned without any network access.

    This function is never imported or called by tests or CI; it exists for
    interactive/CLI use only.
    """
    cache_file = _cache_path(cache_dir, symbol)
    now = datetime.now(UTC)
    if os.path.exists(cache_file):
        age = now - datetime.fromtimestamp(os.path.getmtime(cache_file), tz=UTC)
        if age < ttl:
            return load_csv(cache_file)

    import yfinance as yf  # lazy: package import must stay offline-safe

    frame = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if frame is None or frame.empty:
        msg = f"no data returned for symbol {symbol!r}"
        raise RuntimeError(msg)
    close = frame["Close"]
    if isinstance(close, pd.DataFrame):  # yfinance multi-ticker shape
        close = close.iloc[:, 0]
    series = pd.Series(
        close.to_numpy(dtype=float), index=pd.DatetimeIndex(close.index), name="close"
    ).dropna()
    os.makedirs(cache_dir, exist_ok=True)
    series.rename_axis("date").to_frame().to_csv(cache_file)
    return series
