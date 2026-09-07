"""Stage 2 evaluation runner: measure the Stage 1 pipeline AS-IS. No tuning.

Runs the unchanged Stage 1 expanding-origin walk-forward harness (same
min_train_rows, same features, same sklearn-default HistGradientBoostingRegressor)
for persistence and hist_gradient_boosting on every committed offline fixture,
then records the documented metrics per fixture and per origin window:

- windows: "full" (all walk-forward origins), "first_half" and "second_half"
  (positional split of the origin-ordered forecast list; see
  stock_prediction.metrics.window_slices).

Outputs (both committed):

- experiments/results.csv : one row per (fixture, window, model).
- experiments/eval_log.md : human-readable log with run date, git commit,
  metric definitions, the same table, and per-fixture notes.

Offline only: fixtures are loaded with data.load_csv; yfinance is never
imported. Run with the project venv from the repo root:

    make eval      # or: .venv/Scripts/python experiments/run_eval.py

This is measurement, not model selection. Nothing here tunes features,
min_train_rows, or GBM hyperparameters. The metrics themselves are
deterministic given the committed fixtures and the pinned sklearn version
(HistGradientBoostingRegressor with defaults and <10k training rows uses no
random validation split); the log's run date and git commit are run metadata.
"""

from __future__ import annotations

import csv
import datetime
import os
import subprocess

from stock_prediction.data import load_csv
from stock_prediction.metrics import all_report_fields, metrics_by_window
from stock_prediction.walkforward import run_all_models

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")

FIXTURES: dict[str, str] = {
    "sample_daily": "original Stage 1 fixture: 300-day synthetic geometric random walk",
    "trending_up": "synthetic random walk with constant positive drift (seed 42)",
    "mean_reverting": "synthetic AR(1) log price pulled to a fixed level, negatively autocorrelated returns (seed 7)",
    "vol_regime_shift": "synthetic zero-drift random walk, volatility 0.005 then 0.02 halfway (seed 11)",
}

MODELS = ("persistence", "hist_gradient_boosting")


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # Run outside a git checkout (or git missing): record that explicitly
        # instead of failing the whole measurement run.
        return "unknown"


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fixture_id in FIXTURES:
        closes = load_csv(os.path.join(FIXTURE_DIR, f"{fixture_id}.csv"))
        results = run_all_models(closes, which="both")
        for result in results:
            reports = metrics_by_window(result)
            for window in ("full", "first_half", "second_half"):
                rows.append(reports[window].as_row(fixture=fixture_id, window=window))
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    path = os.path.join(REPO_ROOT, "experiments", "results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_report_fields())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def _md_table(rows: list[dict[str, object]]) -> str:
    cols = all_report_fields()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows:
        cells = ["" if row[c] is None else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_log(rows: list[dict[str, object]]) -> None:
    commit = git_commit()
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    notes = "\n".join(f"- `{fid}`: {note}" for fid, note in FIXTURES.items())
    text = f"""# Stage 2 evaluation log - persistence vs hist_gradient_boosting

Measured AS-IS through the unchanged Stage 1 walk-forward harness (expanding
origins, min_train_rows=60, Stage 1 features, sklearn-default
HistGradientBoostingRegressor). No tuning of any kind was applied for this run.

- Run date: {stamp}
- Git commit: {commit}
- Harness: expanding-origin walk-forward, one-step-ahead log returns, models fit
  only on rows strictly before each origin (see src/stock_prediction/walkforward.py).
- Fixtures (all committed, offline):
{notes}

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

{_md_table(rows)}

## Notes

- Educational only. All fixtures are synthetic random-walk-style series; these
  numbers are NOT live-market results and do not demonstrate trading ability.
- n is small (219 origins per fixture full window, half that per half window),
  so every number here is noisy, and no significance testing was performed.
- Reading guide: a model only "wins" on edge if its edge beats both
  persistence's 0.0 and the always-long mean_realized_return for the same
  fixture/window.
"""
    path = os.path.join(REPO_ROOT, "experiments", "eval_log.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"wrote {path}")


def print_summary(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["window"] != "full":
            continue
        da = row["directional_accuracy"]
        da_s = "n/a" if da is None else f"{float(da):.3f} (n_dir={row['n_directional']})"
        print(
            f"{row['fixture']:<18} {row['model']:<24} n={row['n']:<4} "
            f"rmse={row['rmse']:.6f} mae={row['mae']:.6f} dir_acc={da_s:<18} "
            f"edge={row['edge']:+.6f} always_long={row['mean_realized_return']:+.6f}"
        )


def main() -> int:
    rows = collect_rows()
    print_summary(rows)
    write_csv(rows)
    write_log(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
