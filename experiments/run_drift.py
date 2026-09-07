"""Stage 5 drift runner: reproduce the recorded fixture drift numbers.

Runs the Stage 5 detector (stock_prediction.drift) on the committed offline
fixtures -- NO model fitting, NO tuning, NO retrain -- and rewrites
experiments/drift_log.md with the comparison table:

- within-fixture: first half vs second half of each fixture;
- cross-fixture: sample_daily (baseline) vs each other fixture.

Offline only; yfinance is never imported. Run with the project venv:

    make drift      # or: .venv/Scripts/python experiments/run_drift.py

The detector configuration (PSI bins, thresholds, window definitions) is
documented in the generated log and in src/stock_prediction/drift.py.
"""

from __future__ import annotations

import datetime
import os
import subprocess

from stock_prediction.data import load_csv
from stock_prediction.drift import (
    KS_ALPHA,
    PSI_RETRAIN_THRESHOLD,
    PSI_STABLE,
    compare_windows,
    detect_drift,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
LOG_PATH = os.path.join(REPO_ROOT, "experiments", "drift_log.md")

FIXTURES: dict[str, str] = {
    "sample_daily": "original Stage 1 fixture: 300-day synthetic geometric random walk",
    "trending_up": "synthetic random walk with constant positive drift (seed 42)",
    "mean_reverting": "synthetic AR(1) log price pulled to a fixed level (seed 7)",
    "vol_regime_shift": "synthetic zero-drift random walk, volatility 0.005 then 0.02 halfway (seed 11)",
}


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
        return "unknown"


def _comparison_rows(report) -> list[str]:
    comp = report.comparisons[0]
    fired = "YES" if comp.fired else "no"
    return [
        (
            f"| {report.expected_label} vs {report.actual_label} "
            f"| {comp.psi:.4f} | {comp.band} "
            f"| {comp.ks_statistic:.4f} | {comp.ks_pvalue:.5f} | {fired} |"
        ),
    ]


def build_log() -> str:
    closes = {name: load_csv(os.path.join(FIXTURE_DIR, f"{name}.csv")) for name in FIXTURES}
    lines = [
        "# Stage 5 drift log (drift detection on committed fixtures)",
        "",
        f"- Run date: {datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Git commit: {git_commit()}",
        (
            "- Detector: `src/stock_prediction/drift.py` -- PSI (5 quantile bins) "
            "+ two-sample KS on the daily log-return distribution."
        ),
        (
            f"- Thresholds: PSI < {PSI_STABLE} stable, {PSI_STABLE}-{PSI_RETRAIN_THRESHOLD} moderate, "
            f">= {PSI_RETRAIN_THRESHOLD} significant (Siddiqi 2006 convention); "
            f"KS alpha {KS_ALPHA}. FIRES when PSI >= {PSI_RETRAIN_THRESHOLD} or KS p <= {KS_ALPHA}; "
            "a fired comparison is the documented signal that WOULD trigger a "
            "retrain. The retrain itself is Stage 6 and is NOT implemented."
        ),
        (
            "- Windows: within-fixture = first half vs second half at the midpoint "
            "row; cross-fixture = full baseline series vs full candidate series. "
            "Log returns are computed within each window."
        ),
        (
            "- Scope: committed SYNTHETIC fixtures only. This is not production "
            "monitoring; there is no alerting, scheduler, or live data here."
        ),
        "",
        "## Within-fixture (first half vs second half)",
        "",
        "| Comparison | PSI | PSI band | KS stat | KS p-value | Fired |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in FIXTURES:
        lines += _comparison_rows(detect_drift(closes[name], label=name))
    lines += [
        "",
        "Notes:",
        "",
        (
            "- `sample_daily`/`trending_up`/`mean_reverting` halves are matched "
            "windows generated with constant parameters: all stay quiet, as they "
            "must."
        ),
        (
            "- `vol_regime_shift` halves are a known shifted pair (sigma doubles "
            "halfway by construction): the detector fires decisively "
            "(PSI 1.13, significant band)."
        ),
        (
            "- `sample_daily` and `trending_up` halves show identical statistics "
            "because the two fixtures share almost the same standardized noise "
            "(return correlation ~ 1.0); trending_up is that noise rescaled to "
            "sigma ~ 0.01 with constant drift added."
        ),
        "",
        "## Cross-fixture (baseline sample_daily vs candidate, full series)",
        "",
        "| Comparison | PSI | PSI band | KS stat | KS p-value | Fired |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in FIXTURES:
        if name == "sample_daily":
            continue
        lines += _comparison_rows(
            compare_windows(
                closes["sample_daily"],
                closes[name],
                expected_label="sample_daily",
                actual_label=name,
            )
        )
    lines += [
        "",
        "Notes:",
        "",
        (
            "- `vs vol_regime_shift`: fires (KS p ~ 0.004, PSI 0.15 moderate; the "
            "vol-shift fixture also contains a quiet half, so its full-series "
            "mixture only differs moderately from the stationary baseline)."
        ),
        (
            "- `vs trending_up`: the documented pure-mean-drift comparison -- the "
            "mean return shifts by +0.0008. It fires on the KS criterion "
            "(p ~ 0.003) while PSI 0.239 lands just below the retrain line "
            "(moderate band): the shift is real but small relative to sigma "
            "~ 0.01-0.014. This fixture ALSO differs in volatility (sigma 0.0093 "
            "vs 0.0140), which is part of why the marginals separate."
        ),
        (
            "- `vs mean_reverting`: stays quiet. The marginal return distributions "
            "are close; the detector compares marginals only and cannot see the "
            "AR(1) autocorrelation difference. Documented limitation."
        ),
        "",
        "## What would trigger a retrain (Stage 6, not implemented)",
        "",
        (
            f"Any comparison above with Fired = YES (PSI >= {PSI_RETRAIN_THRESHOLD} or KS p <= {KS_ALPHA}) "
            "on freshly observed data is the documented signal to retrain. "
            "Nothing in this repository executes that retrain; there is no "
            "scheduler, no model swap, no rollback, and no alerting."
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    log = build_log()
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(log)
    print(f"wrote {LOG_PATH} ({len(log.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
