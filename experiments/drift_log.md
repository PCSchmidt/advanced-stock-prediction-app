# Stage 5 drift log (drift detection on committed fixtures)

- Run date: 2026-09-07 19:59 UTC
- Git commit: c2ecec0
- Detector: `src/stock_prediction/drift.py` -- PSI (5 quantile bins) + two-sample KS on the daily log-return distribution.
- Thresholds: PSI < 0.1 stable, 0.1-0.25 moderate, >= 0.25 significant (Siddiqi 2006 convention); KS alpha 0.01. FIRES when PSI >= 0.25 or KS p <= 0.01; a fired comparison is the documented signal that WOULD trigger a retrain. The retrain itself is Stage 6 and is NOT implemented.
- Windows: within-fixture = first half vs second half at the midpoint row; cross-fixture = full baseline series vs full candidate series. Log returns are computed within each window.
- Scope: committed SYNTHETIC fixtures only. This is not production monitoring; there is no alerting, scheduler, or live data here.

## Within-fixture (first half vs second half)

| Comparison | PSI | PSI band | KS stat | KS p-value | Fired |
| --- | --- | --- | --- | --- | --- |
| sample_daily:first_half vs sample_daily:second_half | 0.0362 | stable | 0.0940 | 0.52783 | no |
| trending_up:first_half vs trending_up:second_half | 0.0362 | stable | 0.0940 | 0.52783 | no |
| mean_reverting:first_half vs mean_reverting:second_half | 0.0170 | stable | 0.0738 | 0.81331 | no |
| vol_regime_shift:first_half vs vol_regime_shift:second_half | 1.1315 | significant | 0.3356 | 0.00000 | YES |

Notes:

- `sample_daily`/`trending_up`/`mean_reverting` halves are matched windows generated with constant parameters: all stay quiet, as they must.
- `vol_regime_shift` halves are a known shifted pair (sigma doubles halfway by construction): the detector fires decisively (PSI 1.13, significant band).
- `sample_daily` and `trending_up` halves show identical statistics because the two fixtures share almost the same standardized noise (return correlation ~ 1.0); trending_up is that noise rescaled to sigma ~ 0.01 with constant drift added.

## Cross-fixture (baseline sample_daily vs candidate, full series)

| Comparison | PSI | PSI band | KS stat | KS p-value | Fired |
| --- | --- | --- | --- | --- | --- |
| sample_daily vs trending_up | 0.2394 | moderate | 0.1472 | 0.00304 | YES |
| sample_daily vs mean_reverting | 0.0659 | stable | 0.0936 | 0.14533 | no |
| sample_daily vs vol_regime_shift | 0.1549 | moderate | 0.1438 | 0.00408 | YES |

Notes:

- `vs vol_regime_shift`: fires (KS p ~ 0.004, PSI 0.15 moderate; the vol-shift fixture also contains a quiet half, so its full-series mixture only differs moderately from the stationary baseline).
- `vs trending_up`: the documented pure-mean-drift comparison -- the mean return shifts by +0.0008. It fires on the KS criterion (p ~ 0.003) while PSI 0.239 lands just below the retrain line (moderate band): the shift is real but small relative to sigma ~ 0.01-0.014. This fixture ALSO differs in volatility (sigma 0.0093 vs 0.0140), which is part of why the marginals separate.
- `vs mean_reverting`: stays quiet. The marginal return distributions are close; the detector compares marginals only and cannot see the AR(1) autocorrelation difference. Documented limitation.

## What would trigger a retrain (Stage 6, not implemented)

Any comparison above with Fired = YES (PSI >= 0.25 or KS p <= 0.01) on freshly observed data is the documented signal to retrain. Nothing in this repository executes that retrain; there is no scheduler, no model swap, no rollback, and no alerting.
