"""Artifact bundle: store/load the TRAINED primary model with its identity (Stage 3).

A bundle is a directory with exactly two files:

- ``model.joblib``: the fitted model object (joblib.dump of the wrapper in
  ``models.py``, e.g. ``GradientBoostedReturnModel``).
- ``manifest.json``: identity only -- package version, creation time, git
  commit, sklearn version, the feature config the model was trained with
  (lags / rolling windows / ``min_train_rows``), and the identity (name,
  sha256, row count, date range) of the committed fixture it was trained on.

The manifest carries NO metrics and NO performance claims; Stage 2 numbers in
``experiments/`` are evaluation records, not bundle metadata. Artifacts are
gitignored (``artifacts/``); a reviewer either rebuilds by re-running
training (documented in README Operational notes) or reloads from a bundle.

Commands (offline; use a committed fixture):

    python -m stock_prediction.bundle --fixture tests/fixtures/sample_daily.csv --out artifacts/sample_daily
    python -m stock_prediction.bundle --check artifacts/sample_daily

``train_final_model`` fits the primary model ONCE on all usable history
(features + next-step log-return labels). This is the deployment artifact, a
different thing from the walk-forward evaluation harness, which refits per
origin; it does not change any Stage 1/2 model, feature, or hyperparameter.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess

import joblib
import numpy as np
import sklearn

from . import __version__
from .features import FEATURE_NAMES, LAGS, WINDOWS
from .models import GradientBoostedReturnModel, PersistenceModel, get_model
from .walkforward import MIN_TRAIN_ROWS

MODEL_FILE = "model.joblib"
MANIFEST_FILE = "manifest.json"


def git_commit() -> str:
    """Short HEAD hash, or 'unknown' outside a git checkout (never fatal)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _fixture_identity(fixture_path: str | os.PathLike[str]) -> dict[str, object]:
    """Identity of the training data file: name, sha256, rows, date range."""
    from .data import load_csv

    closes = load_csv(fixture_path)
    with open(fixture_path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {
        "name": os.path.basename(str(fixture_path)),
        "sha256": digest,
        "n_rows": len(closes),
        "first_date": str(closes.index.min().date()),
        "last_date": str(closes.index.max().date()),
    }


def feature_config(min_train_rows: int = MIN_TRAIN_ROWS) -> dict[str, object]:
    """The feature/label configuration a bundle was trained under."""
    return {
        "lags": list(LAGS),
        "rolling_windows": list(WINDOWS),
        "feature_names": list(FEATURE_NAMES),
        "min_train_rows": min_train_rows,
        "target": "next-step log return log(close[t]/close[t-1])",
    }


def train_final_model(closes, *, model_name: str = GradientBoostedReturnModel.name):
    """Fit one model on ALL usable rows (deployment artifact, not evaluation).

    Same features, same label, same defaults as everywhere else in this repo;
    nothing is tuned here. The persistence baseline has no parameters, so
    bundling it just stores the class name -- but it is allowed for symmetry.
    """
    from .features import make_frames

    X, y, last_close = make_frames(closes)
    if len(X) == 0:
        msg = "no usable feature rows to train on"
        raise ValueError(msg)
    y_return = np.log(y / last_close).rename("target_return")
    model = get_model(model_name)
    model.fit(X, y_return)
    return model, len(X)


def save_bundle(
    out_dir: str | os.PathLike[str],
    model,
    *,
    fixture_path: str | os.PathLike[str] | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    n_train_rows: int | None = None,
) -> dict[str, object]:
    """Write model.joblib + manifest.json into ``out_dir`` and return the manifest."""
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(str(out_dir), MODEL_FILE))
    manifest: dict[str, object] = {
        "package_version": __version__,
        "created_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "sklearn_version": sklearn.__version__,
        "model_name": model.name,
        "feature_config": feature_config(min_train_rows),
        "training": {"n_train_rows": n_train_rows},
    }
    if fixture_path is not None:
        manifest["fixture"] = _fixture_identity(fixture_path)
    with open(os.path.join(str(out_dir), MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def load_bundle(bundle_dir: str | os.PathLike[str]) -> tuple[object, dict[str, object]]:
    """Load (model, manifest) from a bundle directory.

    Fails loudly if either file is missing or the manifest lacks its identity
    keys, so an incomplete directory can never pass as a bundle.
    """
    model_path = os.path.join(str(bundle_dir), MODEL_FILE)
    manifest_path = os.path.join(str(bundle_dir), MANIFEST_FILE)
    if not os.path.isfile(model_path):
        msg = f"not a bundle: missing {MODEL_FILE} in {bundle_dir!r}"
        raise FileNotFoundError(msg)
    if not os.path.isfile(manifest_path):
        msg = f"not a bundle: missing {MANIFEST_FILE} in {bundle_dir!r}"
        raise FileNotFoundError(msg)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    required = (
        "package_version",
        "created_at",
        "git_commit",
        "sklearn_version",
        "model_name",
        "feature_config",
    )
    missing = [k for k in required if k not in manifest]
    if missing:
        msg = f"manifest missing identity keys: {missing}"
        raise ValueError(msg)
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        msg = f"bundled object {type(model).__name__!r} has no predict()"
        raise TypeError(msg)
    return model, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stock_prediction.bundle",
        description="Save or inspect a trained-model bundle (offline; committed fixtures).",
    )
    parser.add_argument("--fixture", help="committed (date, close) CSV to train on")
    parser.add_argument("--out", help="bundle output directory (e.g. artifacts/sample_daily)")
    parser.add_argument(
        "--model",
        default=GradientBoostedReturnModel.name,
        choices=[GradientBoostedReturnModel.name, PersistenceModel.name],
    )
    parser.add_argument("--check", help="bundle directory to load and describe (no training)")
    args = parser.parse_args(argv)

    if args.check is not None:
        model, manifest = load_bundle(args.check)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        print(f"loaded model={model.name} from {args.check}")
        return 0
    if not args.fixture or not args.out:
        parser.error("either --check DIR, or both --fixture CSV and --out DIR are required")

    from .data import load_csv

    closes = load_csv(args.fixture)
    model, n_train_rows = train_final_model(closes, model_name=args.model)
    manifest = save_bundle(
        args.out,
        model,
        fixture_path=args.fixture,
        n_train_rows=n_train_rows,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"saved bundle: {os.path.join(args.out, MODEL_FILE)} + {MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
