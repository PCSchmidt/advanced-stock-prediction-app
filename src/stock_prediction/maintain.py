"""Stage 6 maintain loop: drift-triggered retrain + model rollback (offline CLI).

This module CLOSES the loop Stage 5 opened: when the drift detector
(``drift.py``) fires (PSI >= 0.25 or KS p <= 0.01 on the daily log-return
distribution), a new model bundle is trained and written; when it stays
quiet, nothing is written. Execution is strictly ON-DEMAND -- a human runs
the command after seeing a fired signal. There is NO scheduler, no cron, no
background process, and no alerting anywhere in this repository.

Commands (all offline, committed fixtures only; from the repo root):

    # drift-gated retrain: detects first, retrains ONLY IF fired, else no-ops
    python -m stock_prediction.maintain --fixture tests/fixtures/vol_regime_shift.csv

    # manual on-demand retrain ignoring the drift gate (explicit override)
    python -m stock_prediction.maintain --fixture tests/fixtures/sample_daily.csv --force

    # point 'current' back to the previous bundle (no retraining)
    python -m stock_prediction.maintain --rollback

    # point 'current' at an explicit bundle directory (validated)
    python -m stock_prediction.maintain --rollback --to v1-vol_regime_shift-20260907T120000Z

    # show the current pointer and the bundle history
    python -m stock_prediction.maintain --status

Versioned bundles (never overwritten in place)
----------------------------------------------
Every retrain writes a NEW directory under ``artifacts/`` named
``v<N>-<fixture-stem>-<UTC timestamp>`` where ``N`` is one more than the
number of valid bundles already present, so existing bundles are immutable.
Bundles are created with the UNCHANGED Stage 3 machinery
(``bundle.train_final_model`` + ``bundle.save_bundle``: same Stage 1 features,
same sklearn-default HistGradientBoosting, same ``min_train_rows``); this
module adds no model, feature, or hyperparameter changes and no new
dependencies.

Pointer mechanics ('current' model)
-----------------------------------
The current model is a one-line pointer file ``artifacts/current`` containing
the bundle directory NAME (not a copy, not a symlink -- Windows-safe).
Retraining moves the pointer forward after the new bundle is validated;
rollback moves it back. Every switch VALIDATES the target bundle's manifest
identity first: the bundle must load (``bundle.load_bundle``), its manifest
must carry the required identity keys, its feature config must equal the
Stage 1 defaults (``bundle.feature_config()``), and its sklearn version must
match the installed one. A bundle that fails validation can never become
current, and a failed switch leaves the previous pointer untouched.

Rollback only flips a pointer -- it never retrains, never rewrites bundle
files, and keeps every bundle on disk so the operation is reversible again.

Honest scope: everything here runs on committed SYNTHETIC fixtures. The
drift gate is the Stage 5 detector with its synthetic-walk-calibrated
thresholds; nothing in this repository retrains automatically, on a
schedule, or on live data.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os

import sklearn

from .bundle import (
    MANIFEST_FILE,
    MODEL_FILE,
    feature_config,
    load_bundle,
    save_bundle,
    train_final_model,
)
from .models import GradientBoostedReturnModel, PersistenceModel
from .walkforward import MIN_TRAIN_ROWS

DEFAULT_ARTIFACTS_DIR = "artifacts"
CURRENT_POINTER = "current"


def _pointer_path(artifacts_dir: str | os.PathLike[str]) -> str:
    return os.path.join(str(artifacts_dir), CURRENT_POINTER)


def current_bundle(artifacts_dir: str | os.PathLike[str] = DEFAULT_ARTIFACTS_DIR) -> str | None:
    """Name of the bundle directory the 'current' pointer names, or None."""
    path = _pointer_path(artifacts_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        name = f.read().strip()
    return name or None


def list_bundles(
    artifacts_dir: str | os.PathLike[str] = DEFAULT_ARTIFACTS_DIR,
) -> list[tuple[str, dict[str, object]]]:
    """(name, manifest) for every bundle under ``artifacts_dir`` with a
    readable manifest carrying a ``created_at``.

    Sorted by ``created_at`` (the order retraining produced them). Directories
    that are not valid bundles are ignored here; the STRICT validation for
    pointer switches is ``validate_bundle_identity``.
    """
    root = str(artifacts_dir)
    if not os.path.isdir(root):
        return []
    found: list[tuple[str, dict[str, object]]] = []
    for name in sorted(os.listdir(root)):
        manifest_path = os.path.join(root, name, MANIFEST_FILE)
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        created = manifest.get("created_at")
        if not isinstance(created, str):
            continue
        found.append((name, manifest))
    found.sort(key=lambda item: str(item[1]["created_at"]))
    return found


def validate_bundle_identity(bundle_dir: str | os.PathLike[str]) -> dict[str, object]:
    """Load a bundle and check its manifest identity; return the manifest.

    Fails loudly (FileNotFoundError / ValueError / TypeError from
    ``load_bundle``, or ValueError here) if the directory is not a valid
    bundle, the manifest lacks identity keys, the feature config is not the
    Stage 1 default, or the sklearn version differs from the installed one.
    """
    _, manifest = load_bundle(bundle_dir)
    expected_cfg = feature_config(min_train_rows=MIN_TRAIN_ROWS)
    if manifest.get("feature_config") != expected_cfg:
        msg = (
            f"bundle {bundle_dir!r} feature_config does not match the Stage 1 "
            "defaults; refusing to point 'current' at an unknown configuration"
        )
        raise ValueError(msg)
    if manifest.get("sklearn_version") != sklearn.__version__:
        msg = (
            f"bundle {bundle_dir!r} was built with scikit-learn "
            f"{manifest.get('sklearn_version')!r} but {sklearn.__version__!r} "
            "is installed; refusing to point 'current' at a mismatched runtime"
        )
        raise ValueError(msg)
    return manifest


def _next_bundle_dir(
    fixture_path: str | os.PathLike[str], artifacts_dir: str | os.PathLike[str]
) -> str:
    """A NEW versioned directory name; never collides with an existing bundle."""
    stem = os.path.splitext(os.path.basename(str(fixture_path)))[0]
    version = len(list_bundles(artifacts_dir)) + 1
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(str(artifacts_dir), f"v{version}-{stem}-{stamp}")


def retrain(
    fixture_path: str | os.PathLike[str],
    artifacts_dir: str | os.PathLike[str] = DEFAULT_ARTIFACTS_DIR,
    *,
    force: bool = False,
    model_name: str = GradientBoostedReturnModel.name,
) -> dict[str, object]:
    """Drift-gated retrain: detect first, write a new bundle ONLY IF fired.

    Runs the Stage 5 detector (``drift.detect_drift``, first half vs second
    half of the fixture). If it does NOT fire and ``force`` is False, nothing
    is written and the result says so. If it fires (or ``force``), the model
    is rebuilt with the unchanged ``bundle.train_final_model`` and saved via
    ``bundle.save_bundle`` into a NEW versioned directory; existing bundles
    are never overwritten. The 'current' pointer then moves to the new bundle
    (the previous pointer, if any, is reported in the result).

    The result dict always carries ``action`` ('retrained' | 'no_op') and is
    JSON-serializable for logging.
    """
    from .data import load_csv
    from .drift import detect_drift

    closes = load_csv(fixture_path)
    report = detect_drift(closes, label=os.path.basename(str(fixture_path)))
    if not force and not report.retrain_recommended:
        return {
            "action": "no_op",
            "fixture": os.path.basename(str(fixture_path)),
            "fired": False,
            "psi": round(report.worst.psi, 6),
            "ks_pvalue": round(report.worst.ks_pvalue, 6),
            "message": (
                "drift detector stayed quiet (PSI < 0.25 and KS p > 0.01); "
                "no bundle written, current pointer unchanged"
            ),
        }

    out_dir = _next_bundle_dir(fixture_path, artifacts_dir)
    model, n_train_rows = train_final_model(closes, model_name=model_name)
    manifest = save_bundle(out_dir, model, fixture_path=fixture_path, n_train_rows=n_train_rows)
    validate_bundle_identity(out_dir)  # the new bundle must pass its own gate

    previous = current_bundle(artifacts_dir)
    with open(_pointer_path(artifacts_dir), "w", encoding="utf-8") as f:
        f.write(os.path.basename(out_dir) + "\n")
    return {
        "action": "retrained",
        "fixture": os.path.basename(str(fixture_path)),
        "fired": report.fired,
        "forced": force,
        "worst_comparison": report.worst.as_dict(),
        "bundle": os.path.basename(out_dir),
        "previous_bundle": previous,
        "manifest": manifest,
    }


def rollback(
    artifacts_dir: str | os.PathLike[str] = DEFAULT_ARTIFACTS_DIR,
    *,
    to: str | None = None,
) -> dict[str, object]:
    """Point 'current' back at a previous bundle. No retraining, no writes
    to any bundle directory.

    With ``to=None`` the target is the bundle immediately BEFORE the current
    one in ``created_at`` order (one step back). With ``to=NAME`` the target
    is that named bundle directory. Either way the target's manifest identity
    is validated FIRST (``validate_bundle_identity``); a failed validation or
    a missing target raises and leaves the pointer untouched.
    """
    history = list_bundles(artifacts_dir)
    if not history:
        msg = f"no bundles found under {artifacts_dir!r}; nothing to roll back to"
        raise FileNotFoundError(msg)
    current = current_bundle(artifacts_dir)

    names = [name for name, _ in history]
    if to is None:
        if current is None:
            msg = (
                "no 'current' pointer set; pass --to <bundle-dir-name> to choose "
                "a target explicitly"
            )
            raise ValueError(msg)
        if current not in names:
            msg = (
                f"current pointer names {current!r} but no such valid bundle "
                f"exists under {artifacts_dir!r}; pass --to explicitly"
            )
            raise ValueError(msg)
        idx = names.index(current)
        if idx == 0:
            msg = f"{current!r} is the oldest bundle; there is no previous version"
            raise ValueError(msg)
        target = names[idx - 1]
    else:
        target = to
        if target not in names:
            msg = f"no valid bundle named {to!r} under {artifacts_dir!r}"
            raise FileNotFoundError(msg)

    target_dir = os.path.join(str(artifacts_dir), target)
    manifest = validate_bundle_identity(target_dir)
    with open(_pointer_path(artifacts_dir), "w", encoding="utf-8") as f:
        f.write(target + "\n")
    return {
        "action": "rolled_back",
        "bundle": target,
        "previous_bundle": current,
        "manifest": manifest,
    }


def status(artifacts_dir: str | os.PathLike[str] = DEFAULT_ARTIFACTS_DIR) -> dict[str, object]:
    """Current pointer + bundle history (name, created_at, model, fixture)."""
    current = current_bundle(artifacts_dir)
    bundles = [
        {
            "bundle": name,
            "created_at": manifest.get("created_at"),
            "model_name": manifest.get("model_name"),
            "fixture": (manifest.get("fixture") or {}).get("name"),
        }
        for name, manifest in list_bundles(artifacts_dir)
    ]
    return {"current": current, "bundles": bundles}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m stock_prediction.maintain",
        description=(
            "Stage 6 maintain loop (offline, on-demand): drift-gated retrain "
            "into a NEW versioned bundle, and rollback of the 'current' "
            "pointer. No scheduler, no background process."
        ),
    )
    parser.add_argument(
        "--fixture",
        help="committed (date, close) CSV to drift-check and, if it fires, retrain on",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="retrain even if the drift detector stays quiet (manual override)",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="point 'current' back to the previous bundle (or --to NAME); no retraining",
    )
    parser.add_argument("--to", help="explicit target bundle directory name for --rollback")
    parser.add_argument("--status", action="store_true", help="show current pointer + history")
    parser.add_argument(
        "--artifacts",
        default=DEFAULT_ARTIFACTS_DIR,
        help="artifacts directory holding bundles and the current pointer "
        f"(default: {DEFAULT_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--model",
        default=GradientBoostedReturnModel.name,
        choices=[GradientBoostedReturnModel.name, PersistenceModel.name],
    )
    args = parser.parse_args(argv)

    if args.status:
        print(json.dumps(status(args.artifacts), indent=2))
        return 0
    if args.rollback:
        result = rollback(args.artifacts, to=args.to)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(f"current -> {result['bundle']} (was {result['previous_bundle']})")
        return 0
    if not args.fixture:
        parser.error("one of --fixture, --rollback, or --status is required")

    result = retrain(args.fixture, args.artifacts, force=args.force, model_name=args.model)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["action"] == "no_op":
        print(result["message"])
    else:
        print(
            f"current -> {result['bundle']} "
            f"(was {result['previous_bundle']}); saved {MODEL_FILE} + {MANIFEST_FILE}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
