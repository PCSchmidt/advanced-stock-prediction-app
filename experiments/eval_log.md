# Stage 2 evaluation log - persistence vs hist_gradient_boosting

Measured AS-IS through the unchanged Stage 1 walk-forward harness (expanding
origins, min_train_rows=60, Stage 1 features, sklearn-default
HistGradientBoostingRegressor). No tuning of any kind was applied for this run.

- Run date: 2026-09-07 16:50 UTC
- Git commit: 5855f51
- Harness: expanding-origin walk-forward, one-step-ahead log returns, models fit
  only on rows strictly before each origin (see src/stock_prediction/walkforward.py).
- Fixtures (all committed, offline):
- `sample_daily`: original Stage 1 fixture: 300-day synthetic geometric random walk
- `trending_up`: synthetic random walk with constant positive drift (seed 42)
- `mean_reverting`: synthetic AR(1) log price pulled to a fixed level, negatively autocorrelated returns (seed 7)
- `vol_regime_shift`: synthetic zero-drift random walk, volatility 0.005 then 0.02 halfway (seed 11)

## Metric definitions

All metrics are on the next-step log return the harness forecasts, i.e.
log(actual_close[t] / last_close[t-1]):

- **rmse / mae**: root mean squared error / mean absolute error between
  predicted and realized next-step log returns (log-return units).
- **directional_accuracy**: over forecasts where the predicted return is
  nonzero AND the realized return is nonzero, the fraction with matching signs.
  Zero predictions take no side and are excluded (`n_zero_predicted` counts
  them; `n_directional` is the scored count). Persistence predicts zero every
  step, so its directional accuracy is undefined (empty cell), not 0.
- **edge** (the single profit/edge proxy): mean(sign(predicted_return) *
  realized_return) - the mean log return per step of a long/flat strategy long
  exactly when the model predicts up, flat otherwise; no costs, no leverage.
  Persistence's edge is exactly 0.0 by construction.
- **mean_realized_return** (context, not the proxy): mean realized log return
  per step = an always-long strategy over the same window.
- **windows**: `full` = all origins; `first_half` / `second_half` = positional
  halves of the origin-ordered forecast list (see metrics.window_slices).

## Results

| fixture | window | model | n | rmse | mae | directional_accuracy | n_directional | n_zero_predicted | edge | mean_realized_return | first_origin | last_origin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sample_daily | full | persistence | 219 | 0.014733 | 0.011528 |  | 0 | 219 | 0.0 | -0.000816 | 2024-04-24 | 2025-02-24 |
| sample_daily | first_half | persistence | 109 | 0.01431 | 0.011506 |  | 0 | 109 | 0.0 | -0.001081 | 2024-04-24 | 2024-09-23 |
| sample_daily | second_half | persistence | 110 | 0.015141 | 0.01155 |  | 0 | 110 | 0.0 | -0.000554 | 2024-09-24 | 2025-02-24 |
| sample_daily | full | hist_gradient_boosting | 219 | 0.016002 | 0.012762 | 0.534247 | 219 | 0 | 0.001375 | -0.000816 | 2024-04-24 | 2025-02-24 |
| sample_daily | first_half | hist_gradient_boosting | 109 | 0.015338 | 0.012243 | 0.53211 | 109 | 0 | 0.00128 | -0.001081 | 2024-04-24 | 2024-09-23 |
| sample_daily | second_half | hist_gradient_boosting | 110 | 0.016635 | 0.013276 | 0.536364 | 110 | 0 | 0.001469 | -0.000554 | 2024-09-24 | 2025-02-24 |
| trending_up | full | persistence | 219 | 0.009808 | 0.007636 |  | 0 | 219 | 0.0 | 0.000122 | 2024-04-24 | 2025-02-24 |
| trending_up | first_half | persistence | 109 | 0.009513 | 0.007615 |  | 0 | 109 | 0.0 | -5.4e-05 | 2024-04-24 | 2024-09-23 |
| trending_up | second_half | persistence | 110 | 0.010091 | 0.007657 |  | 0 | 110 | 0.0 | 0.000297 | 2024-09-24 | 2025-02-24 |
| trending_up | full | hist_gradient_boosting | 219 | 0.010685 | 0.008519 | 0.520548 | 219 | 0 | 0.000721 | 0.000122 | 2024-04-24 | 2025-02-24 |
| trending_up | first_half | hist_gradient_boosting | 109 | 0.010232 | 0.008165 | 0.504587 | 109 | 0 | 0.000537 | -5.4e-05 | 2024-04-24 | 2024-09-23 |
| trending_up | second_half | hist_gradient_boosting | 110 | 0.011116 | 0.008869 | 0.536364 | 110 | 0 | 0.000904 | 0.000297 | 2024-09-24 | 2025-02-24 |
| mean_reverting | full | persistence | 219 | 0.011688 | 0.009096 |  | 0 | 219 | 0.0 | 2.9e-05 | 2024-04-24 | 2025-02-24 |
| mean_reverting | first_half | persistence | 109 | 0.010816 | 0.00843 |  | 0 | 109 | 0.0 | 1.9e-05 | 2024-04-24 | 2024-09-23 |
| mean_reverting | second_half | persistence | 110 | 0.012491 | 0.009755 |  | 0 | 110 | 0.0 | 3.9e-05 | 2024-09-24 | 2025-02-24 |
| mean_reverting | full | hist_gradient_boosting | 219 | 0.013162 | 0.010449 | 0.484018 | 219 | 0 | -0.001023 | 2.9e-05 | 2024-04-24 | 2025-02-24 |
| mean_reverting | first_half | hist_gradient_boosting | 109 | 0.012115 | 0.009519 | 0.504587 | 109 | 0 | 3.6e-05 | 1.9e-05 | 2024-04-24 | 2024-09-23 |
| mean_reverting | second_half | hist_gradient_boosting | 110 | 0.014123 | 0.011372 | 0.463636 | 110 | 0 | -0.002074 | 3.9e-05 | 2024-09-24 | 2025-02-24 |
| vol_regime_shift | full | persistence | 219 | 0.015118 | 0.011254 |  | 0 | 219 | 0.0 | 0.000532 | 2024-04-24 | 2025-02-24 |
| vol_regime_shift | first_half | persistence | 109 | 0.010636 | 0.007288 |  | 0 | 109 | 0.0 | 0.000944 | 2024-04-24 | 2024-09-23 |
| vol_regime_shift | second_half | persistence | 110 | 0.018519 | 0.015184 |  | 0 | 110 | 0.0 | 0.000123 | 2024-09-24 | 2025-02-24 |
| vol_regime_shift | full | hist_gradient_boosting | 219 | 0.016765 | 0.012242 | 0.479452 | 219 | 0 | -0.000739 | 0.000532 | 2024-04-24 | 2025-02-24 |
| vol_regime_shift | first_half | hist_gradient_boosting | 109 | 0.011936 | 0.00779 | 0.486239 | 109 | 0 | -0.000688 | 0.000944 | 2024-04-24 | 2024-09-23 |
| vol_regime_shift | second_half | hist_gradient_boosting | 110 | 0.020454 | 0.016654 | 0.472727 | 110 | 0 | -0.000789 | 0.000123 | 2024-09-24 | 2025-02-24 |

## Notes

- Educational only. All fixtures are synthetic random-walk-style series; these
  numbers are NOT live-market results and do not demonstrate trading ability.
- n is small (219 origins per fixture full window, half that per half window),
  so every number here is noisy, and no significance testing was performed.
- Reading guide: a model only "wins" on edge if its edge beats both
  persistence's 0.0 and the always-long mean_realized_return for the same
  fixture/window.
