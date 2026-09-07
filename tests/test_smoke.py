"""Stage 0 smoke tests: environment sanity and package importability."""

import sys

import stock_prediction


def test_python_version():
    assert sys.version_info >= (3, 11)


def test_package_import():
    assert hasattr(stock_prediction, "__version__")
    assert isinstance(stock_prediction.__version__, str)
    assert stock_prediction.__version__ == "0.1.0"
