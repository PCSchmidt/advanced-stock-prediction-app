"""Stage 0/1 smoke tests: environment sanity and package importability."""

import sys

import pytest

import stock_prediction


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_package_import():
    assert hasattr(stock_prediction, "__version__")
    assert isinstance(stock_prediction.__version__, str)
    assert stock_prediction.__version__ == "0.1.0"


def test_package_import_stays_offline_light():
    """Importing the package must not import yfinance (network-touching)."""
    import importlib

    importlib.reload(stock_prediction)
    assert "yfinance" not in sys.modules
    assert "yfinance" not in stock_prediction.__dict__


def test_cli_offline_on_fixture(capsys):
    """End-to-end smoke: CLI runs walk-forward on the committed fixture offline."""
    import os

    from stock_prediction.cli import main

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_daily.csv")
    rc = main(["--fixture", fixture, "--model", "both"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "model=persistence" in out
    assert "model=hist_gradient_boosting" in out
    assert "walk-forward smoke complete" in out


def test_unknown_model_name_rejected():
    from stock_prediction.walkforward import run_all_models

    with pytest.raises(ValueError, match="unknown selection"):
        run_all_models(load_closes(), which="lstm")


def load_closes():
    import os

    from stock_prediction.data import load_csv

    fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample_daily.csv")
    return load_csv(fixture)
