# advanced_stock_prediction_app

Time-series stock forecasting with drift monitoring and retraining. This is the
forecasting + maintain pillar of a personal AI-engineering portfolio.

## Motivation

The goal is to demonstrate the full AI engineering lifecycle on a financial
forecasting problem, with the maintain stage as the differentiator: how the
model degrades as market data drifts, how degradation is detected, and how the
model is retrained. Financial predictions here are educational only; nothing in
this repo is investment advice.

## Method

Planned method (see ROADMAP.md for the full plan):

- Data: free price data (e.g. yfinance) with caching.
- Models: persistence/naive baseline first, then a simple primary model.
- Evaluation: walk-forward validation with no leakage; RMSE, MAE, directional
  accuracy recorded before any tuning.
- Maintain: drift detection on feature/prediction distributions, scheduled or
  drift-triggered retraining, rollback path.

## Results

Stage 0 scaffold. No forecasting has been implemented yet, so there are no
results to report. This section will record honest, reproducible numbers
(baseline comparison across symbols and time windows) once Stage 1-2 work
exists.

## Limitations

- No forecasting, drift detection, or retraining functionality is implemented.
- This is a personal portfolio project, not production software.
- Stock prediction is inherently uncertain; any future results are for
  education only and must not be treated as investment advice.

## Operational notes

Requirements: Python 3.11+, `make`, and `git`.

```
git clone <repo-url>
cd advanced_stock_prediction_app
make setup   # creates .venv and installs the pinned lockfile
make test    # runs pytest and ruff
```

`make lint` runs ruff check + format check; `make clean` removes caches and the
virtualenv. Dependencies are pinned in `requirements-lock.txt`; CI (GitHub
Actions) runs lint and tests on every push and pull request.
