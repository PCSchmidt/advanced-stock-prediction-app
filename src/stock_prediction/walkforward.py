"""Expanding-origin walk-forward validation harness with no leakage.

For each origin time `t` (each row of the feature frame, after a minimum
training-history warm-up):

- the model is fit ONLY on rows whose target time is strictly before `t`;
- the model predicts row `t` (the next step after the training data);
- the origin rolls forward by one step.

Because row `t`'s features use only closes strictly before `t` (enforced by
`features.py` and tested in `tests/test_no_leakage.py`), and the harness fits
only on earlier rows, no information at or after the origin can reach `fit`.
Labels are next-step log returns (log(close[t]/close[t-1])); `t`'s label uses
`t`'s close, which by construction is never in a training set for origin `t`.
`run_walkforward` accepts any object with `fit(X, y)` and `predict(X)`, so the
persistence baseline and the primary model run through the exact same loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import GradientBoostedReturnModel, PersistenceModel

MIN_TRAIN_ROWS = 60


@dataclass(frozen=True)
class Forecast:
    """One walk-forward prediction at one origin."""

    origin: pd.Timestamp  # target time being predicted
    last_close: float  # close[origin - 1], known at forecast time
    predicted_return: float  # model output (log return)
    predicted_close: float  # last_close * exp(predicted_return)
    actual_close: float  # close[origin], known only after the fact


@dataclass(frozen=True)
class WalkForwardResult:
    model_name: str
    forecasts: tuple[Forecast, ...]

    @property
    def n_forecasts(self) -> int:
        return len(self.forecasts)


def run_walkforward(
    model,
    closes: pd.Series,
    *,
    min_train_rows: int = MIN_TRAIN_ROWS,
) -> WalkForwardResult:
    """Walk-forward a single model over the full fixture history.

    `model` may be the persistence baseline or the primary model; both are fit
    and predicted through this same code path. Fitting never sees rows at or
    after the origin (verified by tests using a recording spy model).
    """
    from .features import make_frames

    X, y, last_close = make_frames(closes)
    if len(X) <= min_train_rows:
        msg = (
            f"not enough usable rows: {len(X)} after warm-up, "
            f"need more than min_train_rows={min_train_rows}"
        )
        raise ValueError(msg)

    # Model contract (see models.py): fit/predict the next-step LOG RETURN,
    # i.e. log(close[t] / close[t-1]) at row t. Labels use close[t], so they
    # are only ever passed to fit for rows strictly before the origin.
    y_return = np.log(y / last_close).rename("target_return")

    forecasts: list[Forecast] = []
    for i in range(min_train_rows, len(X)):
        origin = X.index[i]
        train_X = X.iloc[:i]  # rows with target time strictly before origin
        train_y = y_return.iloc[:i]
        model.fit(train_X, train_y)
        pred_return = float(np.asarray(model.predict(X.iloc[[i]]))[0])
        forecasts.append(
            Forecast(
                origin=origin,
                last_close=float(last_close.loc[origin]),
                predicted_return=pred_return,
                predicted_close=float(last_close.loc[origin]) * float(np.exp(pred_return)),
                actual_close=float(y.loc[origin]),
            )
        )
    return WalkForwardResult(model_name=model.name, forecasts=tuple(forecasts))


def run_all_models(closes: pd.Series, *, which: str = "both") -> list[WalkForwardResult]:
    """Run the selected model(s) through the same walk-forward harness.

    `which` is 'persistence', 'hist_gradient_boosting', or 'both'. No metric
    computation or model comparison happens here (that is Stage 2).
    """
    factories = {
        "persistence": PersistenceModel,
        "hist_gradient_boosting": GradientBoostedReturnModel,
    }
    if which == "both":
        names = ["persistence", "hist_gradient_boosting"]
    elif which in factories:
        names = [which]
    else:
        msg = f"unknown selection {which!r}; use 'persistence', 'hist_gradient_boosting', or 'both'"
        raise ValueError(msg)
    return [run_walkforward(factory(), closes) for name in names for factory in [factories[name]]]
