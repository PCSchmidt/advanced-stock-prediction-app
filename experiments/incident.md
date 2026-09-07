# Stage 6 incident write-up: volatility-regime drift -> retrain -> rollback

Executed for real on this branch (branch `stage6`, commit `7c1f84f`, 2026-09-07,
Windows Git Bash, offline venv). Every command below was actually run and every
output block is verbatim captured output from that run (only the interpreter
path is shortened to `.venv/Scripts/python`). Not a fictional outage narrative.

## Scenario

The deployed model (`current`) was trained on the calm `sample_daily` fixture.
New data arrives in a different volatility regime -- exactly the
`vol_regime_shift` fixture, whose sigma doubles halfway by construction. The
Stage 5 detector fires (PSI 1.13, significant band; KS p < 1e-6). The on-demand
maintain CLI retrains into a NEW v2 bundle and moves the `artifacts/current`
pointer. The v2 manifest identity differs from v1 (new `created_at`, new
fixture sha256) while the feature config, sklearn version, and git commit stay
identical -- proof that only the DATA changed, not the model configuration.
The incident is then "undone" with a rollback, and the pointer's manifest
identity is byte-identical to the pre-incident v1 manifest.

There is no scheduler and no alerting: a human noticed the fired signal and
ran the commands. Everything is offline on committed fixtures.

## Step 1 - Deployed baseline model (pre-incident)

```console
$ .venv/Scripts/python -m stock_prediction.bundle --fixture tests/fixtures/sample_daily.csv --out artifacts/v1-sample_daily-20260907T210350Z
{
  "created_at": "2026-09-07T21:03:52+00:00",
  "feature_config": {
    "feature_names": [
      "ret_lag_1",
      "ret_lag_2",
      "ret_lag_3",
      "ret_lag_4",
      "ret_lag_5",
      "roll_mean_10",
      "roll_mean_20",
      "roll_std_10",
      "roll_std_20"
    ],
    "lags": [
      1,
      2,
      3,
      4,
      5
    ],
    "min_train_rows": 60,
    "rolling_windows": [
      10,
      20
    ],
    "target": "next-step log return log(close[t]/close[t-1])"
  },
  "fixture": {
    "first_date": "2024-01-02",
    "last_date": "2025-02-24",
    "n_rows": 300,
    "name": "sample_daily.csv",
    "sha256": "71e92d420935828768204b3fd1fa4fe1e15115ff6d7f0220364b4770abc756b8"
  },
  "git_commit": "7c1f84f",
  "model_name": "hist_gradient_boosting",
  "package_version": "0.1.0",
  "sklearn_version": "1.9.0",
  "training": {
    "n_train_rows": 279
  }
}
saved bundle: artifacts/v1-sample_daily-20260907T210350Z\model.joblib + manifest.json
```

Exit code: 0.

## Step 2 - Mark the deployed bundle as `current` (pointer only, no retrain)

```console
$ .venv/Scripts/python -m stock_prediction.maintain --rollback --to v1-sample_daily-20260907T210350Z
{
  "action": "rolled_back",
  "bundle": "v1-sample_daily-20260907T210350Z",
  "manifest": {
    "created_at": "2026-09-07T21:03:52+00:00",
    "feature_config": {
      "feature_names": [
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_4",
        "ret_lag_5",
        "roll_mean_10",
        "roll_mean_20",
        "roll_std_10",
        "roll_std_20"
      ],
      "lags": [
        1,
        2,
        3,
        4,
        5
      ],
      "min_train_rows": 60,
      "rolling_windows": [
        10,
        20
      ],
      "target": "next-step log return log(close[t]/close[t-1])"
    },
    "fixture": {
      "first_date": "2024-01-02",
      "last_date": "2025-02-24",
      "n_rows": 300,
      "name": "sample_daily.csv",
      "sha256": "71e92d420935828768204b3fd1fa4fe1e15115ff6d7f0220364b4770abc756b8"
    },
    "git_commit": "7c1f84f",
    "model_name": "hist_gradient_boosting",
    "package_version": "0.1.0",
    "sklearn_version": "1.9.0",
    "training": {
      "n_train_rows": 279
    }
  },
  "previous_bundle": null
}
current -> v1-sample_daily-20260907T210350Z (was None)
```

Exit code: 0.

## Step 3 - Status before the alert

```console
$ .venv/Scripts/python -m stock_prediction.maintain --status
{
  "current": "v1-sample_daily-20260907T210350Z",
  "bundles": [
    {
      "bundle": "v1-sample_daily-20260907T210350Z",
      "created_at": "2026-09-07T21:03:52+00:00",
      "model_name": "hist_gradient_boosting",
      "fixture": "sample_daily.csv"
    }
  ]
}
```

Exit code: 0.

## Step 4 - Drift alert: the Stage 5 detector fires

```console
$ .venv/Scripts/python -m stock_prediction.cli --fixture tests/fixtures/vol_regime_shift.csv --model persistence --drift
model=persistence forecasts=219 first=(2024-04-24 -> 100.1752) last=(2025-02-24 -> 113.3810)
walk-forward smoke complete: all selected models emitted forecasts through the same harness
{
  "expected": "tests/fixtures/vol_regime_shift.csv:first_half",
  "actual": "tests/fixtures/vol_regime_shift.csv:second_half",
  "window_definition": "first half vs second half of one series, split at the midpoint row; daily log returns computed within each half",
  "thresholds": {
    "psi_bins": 5,
    "psi_stable_below": 0.1,
    "psi_retrain_at_or_above": 0.25,
    "ks_alpha": 0.01,
    "rule": "fires when PSI >= 0.25 (significant band) or KS p-value <= 0.01; a fired comparison recommends retrain"
  },
  "fired": true,
  "retrain_recommended": true,
  "worst_feature": {
    "feature": "log_return",
    "n_expected": 149,
    "n_actual": 149,
    "psi": 1.131507,
    "ks_statistic": 0.33557,
    "ks_pvalue": 0.0,
    "band": "significant",
    "fired": true
  },
  "features": [
    {
      "feature": "log_return",
      "n_expected": 149,
      "n_actual": 149,
      "psi": 1.131507,
      "ks_statistic": 0.33557,
      "ks_pvalue": 0.0,
      "band": "significant",
      "fired": true
    }
  ]
}
```

Exit code: 0.

## Step 5 - Maintain: the fired gate triggers a retrain into a NEW v2 bundle

```console
$ .venv/Scripts/python -m stock_prediction.maintain --fixture tests/fixtures/vol_regime_shift.csv
{
  "action": "retrained",
  "bundle": "v2-vol_regime_shift-20260907T210410Z",
  "fired": true,
  "fixture": "vol_regime_shift.csv",
  "forced": false,
  "manifest": {
    "created_at": "2026-09-07T21:04:10+00:00",
    "feature_config": {
      "feature_names": [
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_4",
        "ret_lag_5",
        "roll_mean_10",
        "roll_mean_20",
        "roll_std_10",
        "roll_std_20"
      ],
      "lags": [
        1,
        2,
        3,
        4,
        5
      ],
      "min_train_rows": 60,
      "rolling_windows": [
        10,
        20
      ],
      "target": "next-step log return log(close[t]/close[t-1])"
    },
    "fixture": {
      "first_date": "2024-01-02",
      "last_date": "2025-02-24",
      "n_rows": 300,
      "name": "vol_regime_shift.csv",
      "sha256": "da02f70e335b7e54522fdc494644fa978a2f2e758500f59ea2d9f5912f67efdc"
    },
    "git_commit": "7c1f84f",
    "model_name": "hist_gradient_boosting",
    "package_version": "0.1.0",
    "sklearn_version": "1.9.0",
    "training": {
      "n_train_rows": 279
    }
  },
  "previous_bundle": "v1-sample_daily-20260907T210350Z",
  "worst_comparison": {
    "band": "significant",
    "feature": "log_return",
    "fired": true,
    "ks_pvalue": 0.0,
    "ks_statistic": 0.33557,
    "n_actual": 149,
    "n_expected": 149,
    "psi": 1.131507
  }
}
current -> v2-vol_regime_shift-20260907T210410Z (was v1-sample_daily-20260907T210350Z); saved model.joblib + manifest.json
```

Exit code: 0.

## Step 6 - v2 identity: new `created_at` + new fixture sha256; same feature config / sklearn / git commit

```console
$ .venv/Scripts/python -m stock_prediction.bundle --check artifacts/v2-vol_regime_shift-20260907T210410Z
{
  "created_at": "2026-09-07T21:04:10+00:00",
  "feature_config": {
    "feature_names": [
      "ret_lag_1",
      "ret_lag_2",
      "ret_lag_3",
      "ret_lag_4",
      "ret_lag_5",
      "roll_mean_10",
      "roll_mean_20",
      "roll_std_10",
      "roll_std_20"
    ],
    "lags": [
      1,
      2,
      3,
      4,
      5
    ],
    "min_train_rows": 60,
    "rolling_windows": [
      10,
      20
    ],
    "target": "next-step log return log(close[t]/close[t-1])"
  },
  "fixture": {
    "first_date": "2024-01-02",
    "last_date": "2025-02-24",
    "n_rows": 300,
    "name": "vol_regime_shift.csv",
    "sha256": "da02f70e335b7e54522fdc494644fa978a2f2e758500f59ea2d9f5912f67efdc"
  },
  "git_commit": "7c1f84f",
  "model_name": "hist_gradient_boosting",
  "package_version": "0.1.0",
  "sklearn_version": "1.9.0",
  "training": {
    "n_train_rows": 279
  }
}
loaded model=hist_gradient_boosting from artifacts/v2-vol_regime_shift-20260907T210410Z
```

Exit code: 0.

## Step 7 - Rollback: pointer back to the pre-incident bundle, no retraining

```console
$ .venv/Scripts/python -m stock_prediction.maintain --rollback
{
  "action": "rolled_back",
  "bundle": "v1-sample_daily-20260907T210350Z",
  "manifest": {
    "created_at": "2026-09-07T21:03:52+00:00",
    "feature_config": {
      "feature_names": [
        "ret_lag_1",
        "ret_lag_2",
        "ret_lag_3",
        "ret_lag_4",
        "ret_lag_5",
        "roll_mean_10",
        "roll_mean_20",
        "roll_std_10",
        "roll_std_20"
      ],
      "lags": [
        1,
        2,
        3,
        4,
        5
      ],
      "min_train_rows": 60,
      "rolling_windows": [
        10,
        20
      ],
      "target": "next-step log return log(close[t]/close[t-1])"
    },
    "fixture": {
      "first_date": "2024-01-02",
      "last_date": "2025-02-24",
      "n_rows": 300,
      "name": "sample_daily.csv",
      "sha256": "71e92d420935828768204b3fd1fa4fe1e15115ff6d7f0220364b4770abc756b8"
    },
    "git_commit": "7c1f84f",
    "model_name": "hist_gradient_boosting",
    "package_version": "0.1.0",
    "sklearn_version": "1.9.0",
    "training": {
      "n_train_rows": 279
    }
  },
  "previous_bundle": "v2-vol_regime_shift-20260907T210410Z"
}
current -> v1-sample_daily-20260907T210350Z (was v2-vol_regime_shift-20260907T210410Z)
```

Exit code: 0.

## Step 8 - Identity restored: `current` carries exactly the pre-incident v1 manifest

```console
$ .venv/Scripts/python -m stock_prediction.bundle --check artifacts/v1-sample_daily-20260907T210350Z
{
  "created_at": "2026-09-07T21:03:52+00:00",
  "feature_config": {
    "feature_names": [
      "ret_lag_1",
      "ret_lag_2",
      "ret_lag_3",
      "ret_lag_4",
      "ret_lag_5",
      "roll_mean_10",
      "roll_mean_20",
      "roll_std_10",
      "roll_std_20"
    ],
    "lags": [
      1,
      2,
      3,
      4,
      5
    ],
    "min_train_rows": 60,
    "rolling_windows": [
      10,
      20
    ],
    "target": "next-step log return log(close[t]/close[t-1])"
  },
  "fixture": {
    "first_date": "2024-01-02",
    "last_date": "2025-02-24",
    "n_rows": 300,
    "name": "sample_daily.csv",
    "sha256": "71e92d420935828768204b3fd1fa4fe1e15115ff6d7f0220364b4770abc756b8"
  },
  "git_commit": "7c1f84f",
  "model_name": "hist_gradient_boosting",
  "package_version": "0.1.0",
  "sklearn_version": "1.9.0",
  "training": {
    "n_train_rows": 279
  }
}
loaded model=hist_gradient_boosting from artifacts/v1-sample_daily-20260907T210350Z
```

Exit code: 0.

## Step 9 - Final status: `current` = v1, both bundles kept on disk

```console
$ .venv/Scripts/python -m stock_prediction.maintain --status
{
  "current": "v1-sample_daily-20260907T210350Z",
  "bundles": [
    {
      "bundle": "v1-sample_daily-20260907T210350Z",
      "created_at": "2026-09-07T21:03:52+00:00",
      "model_name": "hist_gradient_boosting",
      "fixture": "sample_daily.csv"
    },
    {
      "bundle": "v2-vol_regime_shift-20260907T210410Z",
      "created_at": "2026-09-07T21:04:10+00:00",
      "model_name": "hist_gradient_boosting",
      "fixture": "vol_regime_shift.csv"
    }
  ]
}
```

Exit code: 0.

## What the identity comparison shows

| Manifest field | v1 (pre-incident) | v2 (post-retrain) | Meaning |
| --- | --- | --- | --- |
| `created_at` | 2026-09-07T21:03:52+00:00 | 2026-09-07T21:04:10+00:00 | new artifact |
| `fixture.name` | sample_daily.csv | vol_regime_shift.csv | retrained on the shifted regime |
| `fixture.sha256` | 71e92d42... | da02f70e... | different training data |
| `git_commit` | 7c1f84f | 7c1f84f | same code |
| `sklearn_version` | 1.9.0 | 1.9.0 | same runtime |
| `feature_config` | Stage 1 defaults | Stage 1 defaults | same features / `min_train_rows`; nothing retuned |
| `training.n_train_rows` | 279 | 279 | same usable history size |

After `--rollback`, `artifacts/current` names `v1-sample_daily-20260907T210350Z`
again, and `bundle --check` on that directory returns exactly the pre-incident
manifest (same `created_at`, same fixture sha256) -- identity restored, no
model was retrained or rewritten during rollback. Both bundles remain on disk
under gitignored `artifacts/`, so rolling forward to v2 again is another
validated pointer flip (`maintain --rollback --to v2-vol_regime_shift-...`).

Honest scope: the "incident" is the committed synthetic `vol_regime_shift`
fixture, not production traffic. The loop (detect -> gate -> retrain into a
new versioned bundle -> validate -> rollback with identity check) is the
implemented claim; alerting, schedulers, and live data are not.
