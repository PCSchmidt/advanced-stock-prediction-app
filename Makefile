# Portable Makefile for Windows (Git Bash) and Linux.
# Detects the venv python path for the current OS.
ifeq ($(OS),Windows_NT)
    VENV_PY := .venv/Scripts/python.exe
else
    VENV_PY := .venv/bin/python
endif

.PHONY: setup test lint format clean

setup:
	python -m venv .venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements-lock.txt

test:
	$(VENV_PY) -m pytest -q
	$(VENV_PY) -m ruff check .

lint:
	$(VENV_PY) -m ruff check .
	$(VENV_PY) -m ruff format --check .

clean:
	rm -rf .venv .pytest_cache .ruff_cache dist build src/*.egg-info
	find . -name __pycache__ -type d -not -path "./legacy/*" -exec rm -rf {} +
	find . -name "*.pyc" -not -path "./legacy/*" -delete
