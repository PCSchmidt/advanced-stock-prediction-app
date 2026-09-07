# ROADMAP.md - advanced-stock-prediction-app

Time-series forecasting application with a strong **maintain** story: drift detection and retraining on changing data. This roadmap drives the build, ship, deploy, monitor, and maintain stages.

## North star

A reviewer can clone this repo, run one command, and see a forecasting app with honest train/test discipline, a baseline it beats, a reproducible deployment, and a documented drift + retraining loop. The README tells the full lifecycle story, with the maintain stage as the standout.

## Stage 0 - Foundation (prereq)

- [x] Inspect existing code and inventory what works vs. what is broken. (See INVENTORY.md.)
- [x] Decide stack (see AGENTS.md defaults) and document the choice. A redesign from JS to Python is acceptable if justified. (Python 3.11+ per AGENTS.md defaults; rationale in INVENTORY.md.)
- [x] Set up Python 3.11+ environment with pinned dependencies and a lockfile.
- [x] Add `.gitignore` for secrets, artifacts, and data caches.
- [x] Establish a test harness and CI (GitHub Actions) that runs tests + lint.
- [x] Write the README skeleton with the Motivation / Method / Results / Limitations / Operational notes structure.

**Acceptance:** `git clone && make setup && make test` succeeds on a clean machine.

## Stage 1 - Build (forecasting core)

- [ ] Data layer: fetch and cache price data from a documented free source.
- [ ] Feature engineering: documented features (returns, lags, rolling stats). Justify choices.
- [ ] Baseline model: persistence/naive forecast as the thing to beat.
- [ ] Primary model: gradient-boosted or lightweight neural approach, kept simple.
- [ ] Walk-forward validation harness with no leakage.

**Acceptance:** Both baseline and primary model produce forecasts through the same validation harness.

## Stage 2 - Evaluate (before optimizing)

- [ ] Metrics: RMSE, MAE, directional accuracy, and a profit/edge proxy if appropriate.
- [ ] Compare primary model vs. baseline across multiple symbols/time windows.
- [ ] Record results in an `experiments/` run log with date, config, and numbers.

**Acceptance:** Baseline comparison is recorded and reproducible. No tuning happens before this.

## Stage 3 - Ship (versioned, reproducible)

- [ ] Model + code versioning: tag releases; pin data and model versions.
- [ ] Artifact bundle: documented way to store and load the trained model + feature config.
- [ ] `requirements.lock` and a reproducible build path.
- [ ] Containerize the app (Dockerfile) and provide `docker-compose.yml`.
- [ ] CI/CD pipeline that builds, tests, and produces a tagged artifact.

**Acceptance:** A tagged release can be rebuilt and run reproducibly.

## Stage 4 - Deploy

- [ ] FastAPI serving layer with health check and a documented API.
- [ ] Deploy target decision: local Docker Compose (minimum) or a public endpoint (optional).
- [ ] Environment-based configuration (no hardcoded secrets).
- [ ] Document the deployment runbook.

**Acceptance:** The app runs from the container and responds to health + forecast endpoints.

## Stage 5 - Monitor (the differentiator)

- [ ] Drift detection: monitor feature/prediction distribution shift over time (e.g., PSI or KS test on rolling windows).
- [ ] Structured logging of requests, forecast latency, and errors.
- [ ] Metrics endpoint exposing: request count, latency percentiles, error rate, drift signal.
- [ ] Optional: Prometheus/Grafana dashboard.
- [ ] Document "what could degrade" (regime change, data source changes, feature drift).

**Acceptance:** A reviewer can see how drift is detected and what signals would trigger a retrain.

## Stage 6 - Maintain

- [ ] Retraining path: scheduled or drift-triggered retrain that rebuilds the model from fresh data.
- [ ] Rollback path: revert to a previous model version.
- [ ] Runbook for common incidents (data source failure, drift alert, degraded accuracy).
- [ ] One documented incident write-up (real or realistic) showing the maintain loop.

**Acceptance:** The maintain loop is documented and executable, not just described.

## Portfolio presentation

- [ ] README tells the full lifecycle story with real numbers, emphasizing the maintain stage.
- [ ] Link the repo from `pcschmidt.github.io`.
- [ ] Prepare a 3-sentence interview arc per lifecycle stage.

## Definition of done

All stages complete, tests green, baseline comparison recorded, deployment reproducible, drift monitoring documented, and the README honestly reflects what is implemented.