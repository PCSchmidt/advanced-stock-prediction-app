"""Models for one-step-ahead return forecasting, sharing one interface.

Both models implement:

    fit(X, y) -> self
    predict(X) -> np.ndarray of predicted *log returns* for each row

The walk-forward harness (`walkforward.py`) converts predicted returns into
forecast prices via `last_close * exp(pred_return)`. All models therefore emit
the same output type through the same harness.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


class PersistenceModel:
    """Naive baseline: predict zero return, i.e. tomorrow's close == today's.

    This is the classic persistence (random-walk) forecast and the thing the
    primary model must beat in Stage 2. It has no parameters, so `fit` stores
    nothing, but it still runs through the identical harness as the primary
    model so comparisons are apples-to-apples.
    """

    name = "persistence"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PersistenceModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class GradientBoostedReturnModel:
    """Primary model: HistGradientBoostingRegressor on past-return features.

    A small gradient-boosted model (sklearn defaults, no tuning in Stage 1).
    Chosen because it is fast, deterministic for fixed data, captures simple
    nonlinear interactions among the lag/rolling features, and stays within
    boring, well-supported tooling (no deep learning). It must beat the
    persistence baseline in Stage 2 before any tuning happens.
    """

    name = "hist_gradient_boosting"

    def __init__(self) -> None:
        self._model = HistGradientBoostingRegressor()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> GradientBoostedReturnModel:
        self._model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict(X.to_numpy(dtype=float))


def get_model(name: str):
    """Return an unfitted model instance by name ('persistence' or
    'hist_gradient_boosting'; 'both' is handled by the harness/CLI, not here)."""
    models = {
        PersistenceModel.name: PersistenceModel,
        GradientBoostedReturnModel.name: GradientBoostedReturnModel,
    }
    if name not in models:
        msg = f"unknown model {name!r}; choose from {sorted(models)}"
        raise ValueError(msg)
    return models[name]()
