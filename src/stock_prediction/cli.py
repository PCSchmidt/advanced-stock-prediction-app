"""Small CLI: run offline walk-forward forecasts on a committed fixture.

Example (from the repo root):

    .venv/Scripts/python -m stock_prediction.cli --fixture tests/fixtures/sample_daily.csv --model both

The CLI prints one smoke line per model (number of walk-forward forecasts and
the first/last forecast). It computes no metrics and makes no comparison -
evaluation is Stage 2. This command is fully offline.
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
    args = parser.parse_args(argv)

    from .data import load_csv  # imported here to keep module import surface small

    closes = load_csv(args.fixture)
    results = run_all_models(closes, which=args.model)
    for result in results:
        first, last = result.forecasts[0], result.forecasts[-1]
        print(
            f"model={result.model_name} forecasts={result.n_forecasts} "
            f"first=({first.origin.date()} -> {first.predicted_close:.4f}) "
            f"last=({last.origin.date()} -> {last.predicted_close:.4f})"
        )
    print(
        "walk-forward smoke complete: all selected models emitted forecasts through the same harness"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
