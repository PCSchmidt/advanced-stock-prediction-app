# AGENTS.md - advanced-stock-prediction-app

Project instructions for AI coding agents working in this repository.

## Purpose

Turn `advanced-stock-prediction-app` into a portfolio-grade time-series forecasting application that demonstrates the full AI engineering lifecycle, with a special emphasis on the **maintain** stage: monitoring for drift and retraining on changing data.

This is a personal portfolio project, not production work. It must be honest, rigorous, and reproducible. Financial predictions are educational only; never present them as investment advice.

## Current state (as of Sep 2026)

- Dormant since Jun 2025.
- Existing functionality: a time-series stock price predictor built as an educational tool.
- JavaScript-based. The user is agnostic to stack, so a redesign is allowed if it serves the goal better.

## Non-negotiable requirements

1. **Real forecasting discipline.** Train/test split with no leakage, walk-forward validation, and a proper baseline (e.g., persistence/naive forecast) to beat.
2. **Drift and maintenance story.** Because financial data changes over time, the repo must show how the model degrades and how it is retrained. This is the differentiator.
3. **Evaluation before optimization.** Record metrics (e.g., RMSE, MAE, directional accuracy) against a baseline before tuning.
4. **Reproducibility.** Pinned dependencies, lockfile, documented setup.
5. **Honest documentation.** Motivation, Method, Results, Limitations, Operational notes. Never claim a capability that is not implemented.
6. **Tests.** A test suite must exist and pass.

## Tech stack guidance

The user is agnostic to stack. Prefer boring, well-supported tools. Python is the default for the forecasting core. Suggested defaults (change only with justification):

- Language: Python 3.11+
- Data: yfinance or a documented free source, with caching
- Modeling: start with a strong baseline (persistence, linear), then optionally a gradient-boosted model or a lightweight neural approach. Do not overcomplicate.
- Serving: FastAPI
- Packaging: Docker + docker-compose
- CI/CD: GitHub Actions
- Monitoring: drift detection (e.g., feature/prediction distribution shift), structured logging, metrics endpoint

## Working conventions

- Keep the model-facing tool surface small. Prefer a persistent Python REPL as the control environment.
- Run project commands from the repo root.
- After any code change, run the test suite and the linter.
- Update ROADMAP.md as work progresses. Mark completed items.
- Use git branches for each lifecycle stage. Commit with clear messages.
- Do not commit secrets, API keys, or large model artifacts.

## Lifecycle stage ownership

This repo owns the **forecasting + drift / retraining (maintain)** pillar of the portfolio. Do not duplicate the RAG or optimization work that belongs to the sibling repos.

## Context files

- `ROADMAP.md` in this directory is the source of truth for planned and completed work.
- Global instructions may also be loaded from `~/.prime/agent/AGENTS.md`; project instructions here take precedence for this repo.