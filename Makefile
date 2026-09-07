# Portable Makefile for Windows (Git Bash) and Linux.
# Detects the venv python path for the current OS.
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.PHONY: setup test eval drift lint format clean

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-lock.txt
	$(VENV_PY) -m pip install -e . --no-deps

test:
	$(VENV_PY) -m pytest -q
	$(VENV_PY) -m ruff check .

# Stage 2 evaluation: persistence vs hist_gradient_boosting on all committed
# fixtures, through the unchanged Stage 1 walk-forward harness. Offline, but
# slower than test (a full GBM walk-forward per fixture), so it is NOT part of
# `make test` / CI. Rewrites experiments/results.csv and experiments/eval_log.md.
eval:
	$(VENV_PY) experiments/run_eval.py

# Stage 5 drift log: run the read-only drift detector on the committed
# fixtures and rewrite experiments/drift_log.md. Offline and fast (no model
# fitting); NOT part of `make test` / CI.
drift:
	$(VENV_PY) experiments/run_drift.py

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m ruff format --check .

clean:
	rm -rf .venv .pytest_cache .ruff_cache dist build src/*.egg-info
	find . -name __pycache__ -type d -not -path "./legacy/*" -exec rm -rf {} +
	find . -name "*.pyc" -not -path "./legacy/*" -delete
