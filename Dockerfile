# Local build/run image for the offline forecasting pipeline (Stage 3).
# Base python version is a build arg so the Python floor is verifiable:
#   docker build --build-arg PYTHON_VERSION=3.11 .   # fails: numpy 2.5.3 needs >=3.12
#   docker build .                                   # default 3.12
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

# Dependencies come ONLY from requirements-lock.txt (the one lockfile).
COPY requirements-lock.txt ./
RUN pip install --no-cache-dir -r requirements-lock.txt

# Then the package itself (no deps: the lock already pinned them).
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Committed offline fixtures: `docker compose up` runs the CLI on one of these
# and never imports the lazy yfinance fetch path, so no network is needed.
COPY tests/fixtures ./tests/fixtures

# Phase 3: committed Stage 2 evaluation artifact for the EVALUATION-CONTEXT
# Prometheus gauges (stock_prediction_eval_*). The api compose service points
# STOCK_PREDICTION_EVAL_RESULTS at this path; without it the gauges would
# stay absent (documented, non-fatal).
COPY experiments/results.csv ./experiments/results.csv

CMD ["python", "-m", "stock_prediction.cli", "--fixture", "tests/fixtures/sample_daily.csv", "--model", "both"]
