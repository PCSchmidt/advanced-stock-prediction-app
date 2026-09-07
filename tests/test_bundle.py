"""Stage 3 bundle tests: save/load roundtrip and manifest identity (offline).

The roundtrip uses a short in-memory series (fast) and the committed fixture
for data-identity fields. No network, no yfinance, no docker.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
import pytest
import sklearn

import stock_prediction
from stock_prediction.bundle import (
    MANIFEST_FILE,
    MODEL_FILE,
    feature_config,
    load_bundle,
    save_bundle,
    train_final_model,
)
from stock_prediction.models import GradientBoostedReturnModel
from stock_prediction.walkforward import MIN_TRAIN_ROWS


def short_closes(n: int = 110) -> pd.Series:
    rng = np.random.default_rng(123)
    rets = rng.normal(0.0002, 0.01, size=n)
    return pd.Series(
        100.0 * np.exp(np.cumsum(rets)),
        index=pd.bdate_range("2024-01-02", periods=n),
        name="close",
    )


def fixture_path() -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures", "sample_daily.csv")


def test_feature_config_matches_stage1_defaults():
    cfg = feature_config()
    assert cfg["lags"] == [1, 2, 3, 4, 5]
    assert cfg["rolling_windows"] == [10, 20]
    assert cfg["min_train_rows"] == MIN_TRAIN_ROWS
    assert "ret_lag_1" in cfg["feature_names"] and "roll_std_20" in cfg["feature_names"]


def test_save_load_roundtrip_predicts_identically(tmp_path):
    model, n_rows = train_final_model(short_closes())
    manifest = save_bundle(tmp_path, model, n_train_rows=n_rows)

    assert os.path.isfile(os.path.join(str(tmp_path), MODEL_FILE))
    assert os.path.isfile(os.path.join(str(tmp_path), MANIFEST_FILE))
    assert manifest["training"]["n_train_rows"] == n_rows > 0

    loaded, loaded_manifest = load_bundle(tmp_path)
    assert loaded_manifest == manifest
    assert isinstance(loaded, GradientBoostedReturnModel)

    from stock_prediction.features import make_frames

    X, _, _ = make_frames(short_closes())
    np.testing.assert_allclose(loaded.predict(X), model.predict(X), rtol=0, atol=0)


def test_manifest_identity_fields(tmp_path):
    model, _ = train_final_model(short_closes())
    manifest = save_bundle(tmp_path, model)
    assert manifest["package_version"] == stock_prediction.__version__
    assert manifest["sklearn_version"] == sklearn.__version__
    assert manifest["model_name"] == "hist_gradient_boosting"
    assert isinstance(manifest["git_commit"], str) and manifest["git_commit"]
    assert "created_at" in manifest
    assert "metrics" not in manifest  # identity only; no performance claims


def test_manifest_records_fixture_identity(tmp_path):
    model, _ = train_final_model(short_closes())
    save_bundle(tmp_path, model, fixture_path=fixture_path())
    with open(os.path.join(str(tmp_path), MANIFEST_FILE), encoding="utf-8") as f:
        on_disk = json.load(f)
    fixture = on_disk["fixture"]
    assert fixture["name"] == "sample_daily.csv"
    with open(fixture_path(), "rb") as f:
        assert fixture["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert fixture["n_rows"] == 300
    assert fixture["first_date"] == "2024-01-02" and fixture["last_date"] == "2025-02-24"


def test_load_bundle_rejects_incomplete_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a bundle"):
        load_bundle(tmp_path)
    (tmp_path / MODEL_FILE).write_bytes(b"junk")
    with pytest.raises(FileNotFoundError, match=MANIFEST_FILE):
        load_bundle(tmp_path)


def test_bundle_cli_train_and_check(tmp_path, capsys):
    from stock_prediction.bundle import main

    out_dir = str(tmp_path / "bundle")
    rc = main(["--fixture", fixture_path(), "--out", out_dir])
    assert rc == 0
    rc = main(["--check", out_dir])
    out = capsys.readouterr().out
    assert rc == 0
    assert '"model_name": "hist_gradient_boosting"' in out
    assert '"sha256"' in out
    assert "loaded model=hist_gradient_boosting" in out
