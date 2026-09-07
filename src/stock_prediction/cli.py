"""Small CLI: run offline walk-forward forecasts on a committed fixture.

Example (from the repo root):

    .venv/Scripts/python -m stock_prediction.cli --fixture tests/fixtures/sample_daily.csv --model both

The CLI prints one smoke line per model (number of walk-forward forecasts and
the first/last forecast). It computes no metrics and makes no comparison -
evaluation is Stage 2. This command is fully offline.

Stage 5 options:

- `--drift` runs the drift detector (drift.py, first half vs second half of
  the fixture) after the smoke and prints the report as JSON. It only READS
  closes; nothing here is retrained -- the drift-gated retrain is the
  separate `python -m stock_prediction.maintain` command (Stage 6).
- `--json-logs` emits the smoke events as structured JSON log lines on stdout
  (obs.py) in addition to the human-readable lines, so the same logging layer
  serves CLI and uvicorn contexts.
"""

from __future__ import annotations

import argparse

from .walkforward import run_all_models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stock_prediction.cli",
        description="Offline walk-forward forecast smoke on a committed fixture CSV.",
    )
    parser.add_argument("--fixture", required=True, help="path to a (date, close) CSV fixture")
    parser.add_argument(
        "--model",
        default="both",
        choices=["persistence", "hist_gradient_boosting", "both"],
        help="which model(s) to run through the harness",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="after the smoke, run the Stage 5 drift detector on this fixture "
        "(first half vs second half) and print the report as JSON",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="also emit the smoke events as structured JSON log lines on stdout",
    )
    args = parser.parse_args(argv)

    from .data import load_csv  # imported here to keep module import surface small

    if args.json_logs:
        from .obs import configure_logging, log_event

        configure_logging()

    closes = load_csv(args.fixture)
    results = run_all_models(closes, which=args.model)
    for result in results:
        first, last = result.forecasts[0], result.forecasts[-1]
        print(
            f"model={result.model_name} forecasts={result.n_forecasts} "
            f"first=({first.origin.date()} -> {first.predicted_close:.4f}) "
            f"last=({last.origin.date()} -> {last.predicted_close:.4f})"
        )
        if args.json_logs:
            log_event(
                "forecast_smoke",
                model=result.model_name,
                n_forecasts=result.n_forecasts,
            )
    print(
        "walk-forward smoke complete: all selected models emitted forecasts through the same harness"
    )
    if args.json_logs:
        log_event("forecast_smoke_complete", model=args.model)

    if args.drift:
        import json

        from .drift import detect_drift

        report = detect_drift(closes, label=args.fixture)
        print(json.dumps(report.as_dict(), indent=2))
        if args.json_logs:
            log_event(
                "drift_check",
                drift_signal=report.fired,
                psi=report.worst.psi,
                fixture=args.fixture,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
