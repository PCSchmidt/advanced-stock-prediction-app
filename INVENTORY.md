> **Historical document (2026-09-08):** the `legacy/` snapshot this inventory describes was removed from the working tree after the redesign was published. Its contents remain recoverable from this repository's git history. The audit below is preserved as the Stage 0 record.

# INVENTORY.md - legacy/ reference audit

Audit of `legacy/` (the original GitHub app, dormant since Jun 2025) against
`AGENTS.md` requirements and `ROADMAP.md` items. `legacy/` is read-only
reference material; nothing here is shipped. A redesign from JS + heavy
statistical stack to a boring Python core is explicitly allowed by AGENTS.md.

## legacy/ contents

| Path | What it is | Notes vs. AGENTS.md / ROADMAP.md |
| --- | --- | --- |
| `main.py` | FastAPI backend: `/predict/{symbol}` endpoint. Fetches prices via `yfinance`, fits ARIMA / SARIMA / GARCH / LSTM, returns 30-day forecast. In-memory 1-hour result cache. | Closest legacy analog to the Stage 1 data layer and Stage 4 serving layer. Reusable *ideas*: yfinance fetch + cache, FastAPI surface. Not reusable as-is (see gaps). |
| `app.py` | Flask WSGI shim that serves `frontend/` and mounts the FastAPI app under `/api` (for Heroku/gunicorn). | Deployment artifact only. ROADMAP Stage 4 wants Docker + FastAPI; this Flask wrapper will be dropped. |
| `Procfile` | Heroku dyno command (`gunicorn app:app`). | Heroku deploy path; replaced by Docker/Compose (ROADMAP Stage 3-4). |
| `runtime.txt` | `python-3.9.16` (Heroku buildpack pin). | Below the Python 3.11+ requirement. Confirms a stack refresh is needed. |
| `requirements.txt` | Pinned deps: fastapi, uvicorn, yfinance, pandas, statsmodels, arch, scikit-learn, tensorflow-cpu/keras, pmdarima, matplotlib, Flask, gunicorn. | No lockfile, mixed pin styles, heavy stack (TensorFlow) that AGENTS.md says not to overcomplicate with. Superseded by `pyproject.toml` + `requirements-lock.txt`. |
| `README.md` | Feature/deploy docs (Heroku section appended). | Good model-choice prose, but no evaluation results, no baseline, no limitations section. Replaced by the Motivation/Method/Results/Limitations/Operational-notes skeleton. |
| `frontend/` (`index.html`, `script.js`, `styles.css`, `images/backdrop.jpg`) | Static HTML/CSS/JS UI with Chart.js, calls the FastAPI endpoint. | AGENTS.md: user is stack-agnostic; redesign allowed. No front-end exists in Stage 0 scope; future serving is API-first (FastAPI). |
| `.gitignore`, `.slugignore` | Standard Python + Heroku ignores. | Patterns absorbed into the new root `.gitignore`. |

## What legacy does well

- Free data source (yfinance) with caching — matches AGENTS.md data guidance.
- FastAPI backend — matches the suggested serving stack.
- Multiple model choices with basic error handling and logging.

## Gaps AGENTS.md explicitly calls out (absent in legacy)

1. **No forecasting discipline.** No train/test split, no walk-forward validation,
   no persistence/naive baseline. Models are fit on all history and forecast
   30 days straight — this is leakage-adjacent by construction.
2. **No evaluation before optimization.** No RMSE/MAE/directional accuracy
   anywhere; model selection is AIC-based (ARIMA/SARIMA) with no skill
   comparison against a baseline.
3. **No drift/retrain story.** Nothing monitors degradation over time, despite
   this being the app's differentiator (ROADMAP Stage 5-6).
4. **No tests.** The legacy README's own "Future Enhancements" admits it: "Add
   unit tests and integration tests" is still open.
5. **No reproducibility.** No lockfile (requirements.txt is not a lock), no
   CI, Python 3.9 pin, Heroku-only deploy path.
6. **Honesty gaps.** README claims "sophisticated" predictions with no
   recorded results; no Limitations section.
7. **Heavy, dated stack.** tensorflow-cpu 2.12 / keras 2.12 pinned for an
   LSTM that a gradient-boosted or linear model likely matches here
   (AGENTS.md: "do not overcomplicate").

## Mapping to ROADMAP

- Stage 0 (this scaffold): replaces legacy packaging with pyproject + lock,
  adds tests + CI + Makefile + honest README skeleton. Legacy stays untouched
  under `legacy/`.
- Stage 1: legacy `main.py` yfinance usage is the reference for the new data
  layer; modeling starts over with baseline-first discipline.
- Stage 3-4: legacy Heroku artifacts (Procfile, app.py, runtime.txt) are
  superseded by Docker/Compose; no salvage.
- Stage 5-6: nothing in legacy helps; this is the greenfield differentiator.
