# advanced_stock_prediction_app

Time-series stock forecasting with drift monitoring and retraining. This is the
forecasting + maintain pillar of a personal AI-engineering portfolio.

## Motivation

The goal is to demonstrate the full AI engineering lifecycle on a financial
forecasting problem, with the maintain stage as the differentiator: how the
model degrades as market data drifts, how degradation is detected, and how the
model is retrained.

**This project is educational only. Nothing in this repository is investment
advice, and no forecast here should be traded on.**

## Method

Stage 1 (forecasting core) is implemented:

- **Data.** Daily close prices from Yahoo Finance via `yfinance` (free,
  documented source). Live fetching is optional and cached: `fetch_prices`
  writes a CSV under `data/cache/` (gitignored, 24 h TTL) and reads it back
  without network access while fresh. The `yfinance` import is lazy, so tests
  and CI never touch the network. All committed evaluation runs use
  `tests/fixtures/sample_daily.csv`, a small (300-row) synthetic geometric
  random walk committed as an offline fixture.

- **Features (no lookahead).** Features for target time `t` are computed only
  from closes strictly before `t`: the last five daily log returns
  (`ret_lag_1..5`), and rolling mean and standard deviation of daily log
  returns over the previous 10 and 20 days (`roll_mean_10`, `roll_std_10`,
  `roll_mean_20`, `roll_std_20`), all shifted so they end at `t-1`. Returns
  are used because raw prices are non-stationary; lags carry short-term
  momentum; rolling stats summarize recent trend and volatility. A dedicated
  leak-detection test perturbs the future tail of the series and asserts
  earlier feature rows are unchanged (see Limitations for what this does and
  does not prove).

- **Baseline.** A persistence (random-walk) forecast: tomorrow's predicted
  return is zero, so the forecast close equals the last known close. It is
  implemented as a model object and runs through the identical harness as the
  primary model. It is the thing to beat in Stage 2.

- **Primary model.** `sklearn.ensemble.HistGradientBoostingRegressor` (sklearn
  defaults, no tuning) over the feature matrix above, predicting the next daily
  log return; the harness converts the predicted return to a forecast close via
  `last_close * exp(pred_return)`. Chosen because it is fast, deterministic for
  fixed data, captures simple nonlinear interactions among the lag/rolling
  features, and stays in boring, well-supported tooling. No deep learning, no
  fit-on-all-history-then-forecast-30-days (the legacy anti-pattern): this is
  strict one-step-ahead forecasting.

- **Walk-forward validation.** Expanding-origin walk-forward: at each origin
  `t` the model fits only on rows whose target time is strictly before `t`,
  predicts step `t`, then rolls forward one step. The harness is model-agnostic
  (any object with `fit`/`predict`), so baseline and primary model share the
  exact same loop. A second leak-detection test uses a spy model that records
  every `fit` call and asserts no training target is ever at or after the
  origin, for every origin in the run.

Run it offline with the committed fixture:

```
python -m stock_prediction.cli --fixture tests/fixtures/sample_daily.csv --model both
```

## Results

Stage 2 measured the Stage 1 pipeline as-is (no tuning). Both models ran through
the same expanding-origin walk-forward harness on four committed offline
fixtures (the original 300-day synthetic random walk plus three more synthetic
regimes: trending, mean-reverting, volatility-regime-shift), scoring 219
walk-forward origins per fixture (full window), split into first-half and
second-half windows. Metrics are on the next-step log return the harness
forecasts; full tables and definitions are in `experiments/eval_log.md` and
`experiments/results.csv` (regenerate with `make eval`).

Full-window results (RMSE/MAE in log-return units; dir. acc. = directional
accuracy over side-taking forecasts; edge = mean(sign(predicted return) *
realized return) of a long/flat strategy; always-long = mean realized return,
context only):

| Fixture | Model | n | RMSE | MAE | Dir. acc. | Edge | Always-long |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sample_daily | persistence | 219 | 0.014733 | 0.011528 | n/a | 0.0 | -0.000816 |
| sample_daily | hist_gradient_boosting | 219 | 0.016002 | 0.012762 | 0.534 | +0.001375 | -0.000816 |
| trending_up | persistence | 219 | 0.009808 | 0.007636 | n/a | 0.0 | +0.000122 |
| trending_up | hist_gradient_boosting | 219 | 0.010685 | 0.008519 | 0.521 | +0.000721 | +0.000122 |
| mean_reverting | persistence | 219 | 0.011688 | 0.009096 | n/a | 0.0 | +0.000029 |
| mean_reverting | hist_gradient_boosting | 219 | 0.013162 | 0.010449 | 0.484 | -0.001023 | +0.000029 |
| vol_regime_shift | persistence | 219 | 0.015118 | 0.011254 | n/a | 0.0 | +0.000532 |
| vol_regime_shift | hist_gradient_boosting | 219 | 0.016765 | 0.012242 | 0.479 | -0.000739 | +0.000532 |

What the numbers say (honestly):

- **The primary model does not beat the baseline on forecast accuracy.**
  Persistence has lower RMSE and MAE than hist_gradient_boosting on all four
  fixtures, in both half-windows as well as the full window.
- **Directional accuracy is a coin flip.** GBM directional accuracy is
  0.479-0.534 across fixtures (persistence takes no side, so its directional
  accuracy is undefined, not zero).
- **The edge proxy is mixed and tiny.** GBM's long/flat edge is positive on
  sample_daily (+0.001375) and trending_up (+0.000721) - the only windows where
  it beats both persistence's 0.0 and always-long - and negative on
  mean_reverting (-0.001023) and vol_regime_shift (-0.000739). Magnitudes are
  a few basis points per step at n=219 with no significance testing.
- This is the expected outcome for one-step-ahead return forecasting on
  synthetic random walks; it is recorded as the Stage 2 baseline comparison,
  not hidden. Any future tuning must beat these recorded numbers.

## Limitations

- No drift detection or retraining is implemented yet; those are Stage 5-6
  work and nothing here should be read as implying they exist.
- All Stage 2 evaluation runs on committed synthetic series (random-walk-style
  fixtures), not real market data. Performance on synthetic walks is not
  market skill and demonstrates nothing about trading ability. Real data
  requires the network fetch path, which CI does not exercise; no live-market
  results are claimed anywhere in this repository.
- n is small: 219 walk-forward origins per fixture (about half that per
  half-window), four fixtures, no significance testing - every recorded number
  is noisy.
- Nothing is tuned: features, `min_train_rows`, and GBM hyperparameters are
  exactly the Stage 1 defaults, and the recorded numbers reflect that as-is
  state.
- The leak-detection tests catch structural leaks (features reading future
  rows, the harness fitting on data at or after the origin). They cannot rule
  out every subtle leakage path.
- One-step-ahead log-return forecasting on daily data has very low achievable
  signal; the recorded metrics should be read with that in mind.
- **Educational only. This is a personal portfolio project, not production
  software, and not investment advice. Nothing here is a live-market claim.**
- Python floor is 3.12, not 3.11: `numpy 2.5.3` in `requirements-lock.txt`
  requires `>=3.12` (verified by a failing `pip install` of the lock on
  `python:3.11-slim`). CI and the Docker image pin 3.12; the lockfile itself
  is unchanged.

## Operational notes

Requirements: Python 3.12+, `make`, and `git`. The floor is 3.12 because the
lockfile's `numpy 2.5.3` requires `>=3.12` (a `python:3.11-slim` install of the
lock fails; verified via `docker build --build-arg PYTHON_VERSION=3.11 .`).

```
git clone <repo-url>
cd advanced_stock_prediction_app
make setup   # creates .venv and installs the pinned lockfile
make test    # runs pytest and ruff
```

`make test` is fully offline: it uses the committed fixture and never imports
the yfinance fetch path. To fetch real prices interactively (network required):

```
python -c "from stock_prediction.data import fetch_prices; s = fetch_prices('AAPL'); print(s.tail())"
```

`make eval` reruns the Stage 2 evaluation (persistence vs
hist_gradient_boosting on all four committed fixtures through the unchanged
walk-forward harness; offline, slower than `make test`, not part of CI) and
rewrites `experiments/results.csv` and `experiments/eval_log.md`.
`make lint` runs ruff check + format check; `make clean` removes caches and the
virtualenv. Dependencies are pinned in `requirements-lock.txt`; CI (GitHub
Actions) runs lint and tests on every push and pull request.

### Model and data versioning (Stage 3)

- **Model identity.** Primary model: `sklearn.ensemble.HistGradientBoostingRegressor`
  with sklearn defaults (no tuning); its version is pinned in
  `requirements-lock.txt` (currently `scikit-learn==1.9.0`). Baseline:
  persistence (parameter-free). Bundle manifests also record the package
  version (`stock_prediction` 0.1.0).
- **Data identity.** The four committed offline fixtures under `tests/fixtures/`
  (`sample_daily` plus the Stage 2 regimes generated by
  `experiments/generate_fixtures.py`: `trending_up` seed 42, `mean_reverting`
  seed 7, `vol_regime_shift` seed 11). A bundle manifest records the training
  fixture's name, sha256, row count, and date range.
- **Tag convention.** `stageN-vX.Y.Z` (e.g. `stage3-v0.1.0`), where `N` is the
  ROADMAP stage and `X.Y.Z` is the package version. Tags are created locally
  at this stage and are not pushed to any remote.

### Artifact bundle (store/load the trained primary model)

A bundle is a directory with `model.joblib` (the fitted model) and
`manifest.json` (identity only: package version, creation time, git commit,
sklearn version, feature config, and fixture identity -- no metrics). Bundles
are written under `artifacts/`, which is gitignored.

```
# save: train the primary model once on the committed fixture's full history
python -m stock_prediction.bundle --fixture tests/fixtures/sample_daily.csv --out artifacts/sample_daily

# load/inspect an existing bundle
python -m stock_prediction.bundle --check artifacts/sample_daily
```

Rebuild instead of reload: re-run the training command above (deterministic
given the fixture and the pinned sklearn version), or re-run the full Stage 2
evaluation with `make eval`. Note the distinction: a bundle holds the final
model fit once on all usable history, while walk-forward (evaluation) refits
per origin; `make eval` reproduces the recorded Stage 2 numbers either way.

### HTTP API (Stage 4)

`src/stock_prediction/app.py` is a small FastAPI + uvicorn service around the
unchanged Stage 1 walk-forward harness and Stage 2 metrics. The same
persistence and HistGradientBoosting models run through the same expanding-
origin loop the CLI uses -- the API adds no models and no tuning.

Endpoints (interactive schema docs at `/docs`):

- `GET /health` -> `{"status": "ok"}` (liveness only; no model work).
- `POST /forecast` -> runs the real walk-forward harness and returns, per
  model: `n_forecasts`, Stage 2 metrics (rmse, mae, directional_accuracy,
  edge), first/last origin, the final forecast, and the last `k` forecasts
  (`last_k`, default 5). The response is a summary -- it never returns all
  219 full-fixture forecasts.

Request body (all fields optional): `fixture` (name of a committed fixture,
default `sample_daily`), `model` (`persistence` | `hist_gradient_boosting` |
`both`, default `both`), `max_rows` (run on only the last N rows of the
fixture for a bounded, faster harness run; the models are unchanged),
`last_k` (1-50, default 5), `source` (`fixture` default | `live`),
`symbol` (required iff `source: "live"`).

Status codes: `200` success; `400` unknown fixture name, missing `symbol`
with `source: "live"`, or a harness error (not enough usable rows); `403`
`source: "live"` requested while the server was not started with
`ALLOW_LIVE_DATA=1`; `422` schema violations (wrong types, unknown `model`
value, `max_rows` below the walk-forward warm-up minimum); `500` unexpected
server-side failure (e.g. misconfigured fixture directory).

The default path is fully offline: committed fixtures, no network, no keys.
The live yfinance option is double-gated -- the request must ask for
`"source": "live"` AND the server must be started with `ALLOW_LIVE_DATA=1`.

Environment variables (all optional, no secrets; defaults keep everything
offline):

| Variable | Default | Meaning |
| --- | --- | --- |
| `STOCK_PREDICTION_FIXTURE_DIR` | `<repo>/tests/fixtures` (source layout) | Directory holding the committed fixture CSVs. docker-compose sets it to `/app/tests/fixtures` because the package is installed into site-packages in the image. |
| `ALLOW_LIVE_DATA` | unset (live disabled) | Set to `1` to allow `POST /forecast` with `source: "live"` (yfinance fetch; network required). Unset in the container, so the compose service is fixture-only. |

### Deploy target decision

Local Docker Compose is the deploy target: it is the reproducible minimum and
matches the portfolio goal (a reviewer can run it), while a public endpoint
(ngrok, Azure, AWS, or any paid hosting) was **declined** -- recurring cost and
operational surface for zero portfolio value, and this app is explicitly not
production. There is no TLS, no authentication, and no multi-user serving
anywhere in this repository, and none is claimed.

### Deployment runbook (local Docker Compose, verified end-to-end)

Prerequisites: Docker Desktop running, `git`, (optional) `make` + Python 3.12+
for the host-side test suite.

```
git clone <repo-url>
cd advanced_stock_prediction_app
# optional host-side check (offline, docker-free):
make setup && make test

docker compose build api
docker compose up -d api
curl -s --noproxy "*" http://127.0.0.1:8000/health
curl -s --noproxy "*" -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  -d "{"model": "both", "max_rows": 120, "last_k": 3}"
docker compose down
```

Verified on this branch (2026-09-07, Docker 29.7.2, image
`stock-prediction:local`, 833MB): `/health` returned `{"status": "ok"}`
(HTTP 200); `/forecast` with the body above returned HTTP 200 with both
models at 39 walk-forward origins over the last 120 fixture rows; the
default body (`{}`) returned both models at the full 219 origins; an unknown
fixture name returned HTTP 400; `source: "live"` in the container returned
HTTP 403 (env gate off); `docker compose down` stopped and removed the
container cleanly.

Windows Git Bash notes: `curl` ships with Git Bash. Use `--noproxy "*"` if a
proxy env var would otherwise intercept localhost; single-quote the JSON body
(`-d '{"model": "both"}'`) since Git Bash handles single quotes cleanly (the
double-quoted `\"` form above is the portable equivalent); use forward
slashes in paths (`cd c:/Dev/...`).

### Docker / docker-compose (local only)

```
docker build -t stock-prediction:local .
docker compose up -d api                      # Stage 4 HTTP service (fixture-only)
docker compose up --abort-on-container-exit   # Stage 3 offline smokes
docker compose down
```

The image is `python:3.12-slim` and installs only from
`requirements-lock.txt` plus the package. All three compose services stay
offline on committed fixtures: the walk-forward CLI (`forecast-smoke`), a
bundle save/load roundtrip (`bundle-smoke`), and the FastAPI service (`api`,
published on `127.0.0.1:8000`). The lazy yfinance fetch path is never
imported in the default compose configuration and no API keys are involved.
`ALLOW_LIVE_DATA` is intentionally unset in the container: the compose service
serves fixtures only. Nothing is pushed to any registry.
