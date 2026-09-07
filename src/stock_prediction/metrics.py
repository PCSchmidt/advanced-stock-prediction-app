"""Evaluation metrics for walk-forward forecasts (Stage 2).

All metrics are computed on the *next-step log return* the harness forecasts
(see `walkforward.py`): the realized return at origin `t` is
`log(actual_close[t] / last_close[t-1])`, exactly the label models fit on.

Definitions (also documented in README Results and experiments/eval_log.md):

- `rmse`: root mean squared error between predicted and realized next-step log
  returns, in log-return units.
- `mae`: mean absolute error on the same quantity.
- `directional_accuracy`: over forecasts where the predicted return is nonzero
  AND the realized return is nonzero, the fraction where `sign(predicted) ==
  sign(realized)`. Forecasts with zero predicted return (persistence always)
  take no side and are excluded; `n_zero_predicted` reports how many. If no
  forecast takes a side, directional accuracy is `None` (undefined), not 0.
- `edge` (the single profit/edge proxy): `mean(sign(predicted_return) *
  realized_return)` -- the mean realized log return per step of a long/flat
  strategy that is long exactly when the model predicts a positive return and
  flat otherwise (no costs, no leverage). Persistence predicts zero every step,
  so its edge is exactly 0.0 by construction; a model must beat 0 AND the
  always-long mean below to show any synthetic edge.
- `mean_realized_return` (context only, not the proxy): mean realized log
  return per step, i.e. the return of an always-long strategy over the window.

No model fitting, tuning, or feature change happens here; metrics only read the
forecasts the Stage 1 harness already produced.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .walkforward import Forecast, WalkForwardResult


@dataclass(frozen=True)
class MetricReport:
    """Metrics over one window (subset) of walk-forward forecasts."""

    model_name: str
    n: int
    rmse: float
    mae: float
    directional_accuracy: float | None
    n_directional: int
    n_zero_predicted: int
    edge: float
    mean_realized_return: float
    first_origin: pd.Timestamp
    last_origin: pd.Timestamp

    def as_row(self, *, fixture: str, window: str) -> dict[str, object]:
        """Flat dict for the experiments/ results CSV."""
        return {
            "fixture": fixture,
            "window": window,
            "model": self.model_name,
            "n": self.n,
            "rmse": round(self.rmse, 6),
            "mae": round(self.mae, 6),
            "directional_accuracy": (
                None if self.directional_accuracy is None else round(self.directional_accuracy, 6)
            ),
            "n_directional": self.n_directional,
            "n_zero_predicted": self.n_zero_predicted,
            "edge": round(self.edge, 6),
            "mean_realized_return": round(self.mean_realized_return, 6),
            "first_origin": str(self.first_origin.date()),
            "last_origin": str(self.last_origin.date()),
        }


def _sign(x: np.ndarray) -> np.ndarray:
    out = np.sign(x)
    return out


def compute_metrics(model_name: str, forecasts: Sequence[Forecast]) -> MetricReport:
    """Compute the documented metrics over an ordered subset of forecasts."""
    if not forecasts:
        msg = "compute_metrics needs at least one forecast"
        raise ValueError(msg)
    pred = np.array([f.predicted_return for f in forecasts], dtype=float)
    realized = np.array([np.log(f.actual_close / f.last_close) for f in forecasts], dtype=float)
    err = pred - realized
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))

    sign_pred = _sign(pred)
    sign_real = _sign(realized)
    n_zero_predicted = int(np.count_nonzero(sign_pred == 0))
    takes_side = sign_pred != 0
    realized_moves = sign_real != 0
    scored = takes_side & realized_moves
    n_directional = int(np.count_nonzero(scored))
    if n_directional > 0:
        directional_accuracy = float(np.mean(sign_pred[scored] == sign_real[scored]))
    else:
        directional_accuracy = None

    edge = float(np.mean(sign_pred * realized))
    mean_realized = float(np.mean(realized))
    return MetricReport(
        model_name=model_name,
        n=len(forecasts),
        rmse=rmse,
        mae=mae,
        directional_accuracy=directional_accuracy,
        n_directional=n_directional,
        n_zero_predicted=n_zero_predicted,
        edge=edge,
        mean_realized_return=mean_realized,
        first_origin=forecasts[0].origin,
        last_origin=forecasts[-1].origin,
    )


def compute_result_metrics(result: WalkForwardResult) -> MetricReport:
    """Metrics over ALL forecasts of one harness run."""
    return compute_metrics(result.model_name, result.forecasts)


def window_slices(
    forecasts: Sequence[Forecast],
) -> dict[str, list[Forecast]]:
    """Split forecasts into first-half and second-half origin windows.

    The split is positional on the forecast list (origins roll forward one step
    per forecast): `first_half` is `forecasts[:n // 2]`, `second_half` is
    `forecasts[n // 2:]`. Full-window metrics come from the unsplit list.
    """
    n = len(forecasts)
    half = n // 2
    return {
        "full": list(forecasts),
        "first_half": list(forecasts[:half]),
        "second_half": list(forecasts[half:]),
    }


def metrics_by_window(
    result: WalkForwardResult,
) -> dict[str, MetricReport]:
    """Full / first-half / second-half metrics for one harness run."""
    return {
        window: compute_metrics(result.model_name, fs)
        for window, fs in window_slices(result.forecasts).items()
    }


def all_report_fields() -> list[str]:
    """Column order for the results CSV (see MetricReport.as_row)."""
    return [
        "fixture",
        "window",
        "model",
        "n",
        "rmse",
        "mae",
        "directional_accuracy",
        "n_directional",
        "n_zero_predicted",
        "edge",
        "mean_realized_return",
        "first_origin",
        "last_origin",
    ]


def iter_forecasts(forecasts: Iterable[Forecast]) -> tuple[Forecast, ...]:
    return tuple(forecasts)
