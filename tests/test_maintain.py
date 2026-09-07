"""Stage 6 maintain-loop tests: drift-gated retrain, versioned bundles,
rollback with manifest identity validation. Offline, docker-free, fast.

Bundles are written to ``tmp_path`` (never the real ``artifacts/``). The
committed fixtures carry the drift behavior: ``sample_daily`` stays quiet
(matched stationary halves, PSI 0.0362) and ``vol_regime_shift`` fires (sigma
doubles halfway, PSI 1.13). Training is ONE final fit on ~290 rows -- the
same fast path tests/test_bundle.py already exercises.
"""

from __future__ import annotations

import json
import os

import pytest
import sklearn

import stock_prediction
from stock_prediction.bundle import MANIFEST_FILE, MODEL_FILE, feature_config, load_bundle
from stock_prediction.maintain import CURRENT_POINTER, retrain, rollback, status
from stock_prediction.walkforward import MIN_TRAIN_ROWS


def fixture(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures", name)


QUIET = fixture("sample_daily.csv")  # halves stay quiet
FIRED = fixture("vol_regime_shift.csv")  # halves fire (PSI 1.13)


def pointer(artifacts) -> str | None:
    path = os.path.join(str(artifacts), CURRENT_POINTER)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def bundle_dir(result, artifacts) -> str:
    return os.path.join(str(artifacts), result["bundle"])


def test_quiet_fixture_writes_no_bundle(tmp_path):
    result = retrain(QUIET, str(tmp_path))
    assert result["action"] == "no_op"
    assert result["fired"] is False
    assert os.listdir(str(tmp_path)) == []  # nothing written at all
    assert pointer(tmp_path) is None


def test_quiet_with_force_still_retrains(tmp_path):
    result = retrain(QUIET, str(tmp_path), force=True)
    assert result["action"] == "retrained"
    assert result["forced"] is True
    assert os.path.isfile(os.path.join(bundle_dir(result, tmp_path), MODEL_FILE))


def test_fired_fixture_writes_new_bundle_and_moves_pointer(tmp_path):
    result = retrain(FIRED, str(tmp_path))
    assert result["action"] == "retrained"
    assert result["fired"] is True
    assert result["forced"] is False
    assert result["worst_comparison"]["psi"] >= 0.25

    target = bundle_dir(result, tmp_path)
    assert result["bundle"].startswith("v1-vol_regime_shift-")
    assert os.path.isfile(os.path.join(target, MODEL_FILE))
    assert os.path.isfile(os.path.join(target, MANIFEST_FILE))
    assert pointer(tmp_path) == result["bundle"]

    manifest = result["manifest"]
    assert manifest["model_name"] == "hist_gradient_boosting"
    assert manifest["sklearn_version"] == sklearn.__version__
    assert manifest["package_version"] == stock_prediction.__version__
    assert isinstance(manifest["git_commit"], str) and manifest["git_commit"]
    assert manifest["feature_config"] == feature_config(min_train_rows=MIN_TRAIN_ROWS)
    assert manifest["fixture"]["name"] == "vol_regime_shift.csv"
    assert manifest["training"]["n_train_rows"] > 0
    assert "metrics" not in manifest  # identity only, no performance claims


def test_second_retrain_writes_new_directory_never_overwrites(tmp_path):
    first = retrain(FIRED, str(tmp_path))
    second = retrain(FIRED, str(tmp_path), force=True)
    assert first["bundle"] != second["bundle"]
    assert first["bundle"].startswith("v1-")
    assert second["bundle"].startswith("v2-")
    # both bundles still loadable and untouched
    _, m1 = load_bundle(bundle_dir(first, tmp_path))
    _, m2 = load_bundle(bundle_dir(second, tmp_path))
    assert m1["created_at"] <= m2["created_at"]
    assert pointer(tmp_path) == second["bundle"]
    assert second["previous_bundle"] == first["bundle"]


def test_rollback_restores_previous_manifest_identity(tmp_path):
    first = retrain(FIRED, str(tmp_path), force=True)
    second = retrain(FIRED, str(tmp_path), force=True)
    v1, v2 = first["bundle"], second["bundle"]

    result = rollback(str(tmp_path))
    assert result["action"] == "rolled_back"
    assert result["bundle"] == v1
    assert result["previous_bundle"] == v2
    assert pointer(tmp_path) == v1

    # identity restored: the pointed-at bundle IS the v1 manifest, byte-for-byte
    _, manifest_v1 = load_bundle(bundle_dir(first, tmp_path))
    assert result["manifest"] == manifest_v1
    assert manifest_v1 == first["manifest"]
    # rollback rewrote no bundle files; the bundle still loads identically
    _, again = load_bundle(bundle_dir(first, tmp_path))
    assert again == manifest_v1


def test_rollback_forward_again_is_explicit_validated_target(tmp_path):
    retrain(FIRED, str(tmp_path), force=True)
    second = retrain(FIRED, str(tmp_path), force=True)
    rollback(str(tmp_path))
    result = rollback(str(tmp_path), to=second["bundle"])
    assert result["bundle"] == second["bundle"]
    assert pointer(tmp_path) == second["bundle"]


def test_rollback_validates_target_identity(tmp_path):
    first = retrain(FIRED, str(tmp_path), force=True)
    second = retrain(FIRED, str(tmp_path), force=True)
    # corrupt the OLDER bundle's manifest -> rollback must fail LOUDLY
    manifest_path = os.path.join(bundle_dir(first, tmp_path), MANIFEST_FILE)
    with open(manifest_path, encoding="utf-8") as f:
        broken = json.load(f)
    broken.pop("sklearn_version")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(broken, f)
    with pytest.raises(ValueError, match="identity keys"):
        rollback(str(tmp_path))
    # pointer untouched: still the newest bundle
    assert pointer(tmp_path) == second["bundle"]


def test_rollback_errors_without_history_or_previous(tmp_path):
    with pytest.raises(FileNotFoundError):
        rollback(str(tmp_path))  # no bundles at all
    only = retrain(FIRED, str(tmp_path), force=True)
    with pytest.raises(ValueError, match="oldest"):
        rollback(str(tmp_path))  # current is the oldest; nothing before it
    assert pointer(tmp_path) == only["bundle"]


def test_status_reports_pointer_and_history(tmp_path):
    assert status(str(tmp_path)) == {"current": None, "bundles": []}
    first = retrain(FIRED, str(tmp_path), force=True)
    retrain(FIRED, str(tmp_path), force=True)
    snap = status(str(tmp_path))
    assert snap["current"] is not None
    assert len(snap["bundles"]) == 2
    assert [b["bundle"] for b in snap["bundles"]] == [
        first["bundle"],
        snap["current"],
    ]  # created_at order: v1 first
    assert snap["bundles"][0]["fixture"] == "vol_regime_shift.csv"


def test_maintain_cli_retrain_rollback_status(tmp_path, capsys):
    from stock_prediction.maintain import main

    artifacts = str(tmp_path / "artifacts")
    rc = main(["--fixture", FIRED, "--artifacts", artifacts])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"action": "retrained"' in out
    assert '"fired": true' in out

    rc = main(["--fixture", FIRED, "--force", "--artifacts", artifacts])
    capsys.readouterr()
    assert rc == 0

    rc = main(["--rollback", "--artifacts", artifacts])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"action": "rolled_back"' in out
    assert "current -> v1-" in out

    rc = main(["--status", "--artifacts", artifacts])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"current": "v1-' in out
    assert '"fixture": "vol_regime_shift.csv"' in out


def test_maintain_cli_quiet_no_op(tmp_path, capsys):
    from stock_prediction.maintain import main

    rc = main(["--fixture", QUIET, "--artifacts", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"action": "no_op"' in out
    assert "no bundle written" in out
    assert os.listdir(str(tmp_path)) == []
