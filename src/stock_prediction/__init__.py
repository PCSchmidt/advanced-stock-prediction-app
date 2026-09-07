"""stock_prediction: time-series stock forecasting with drift monitoring and retraining.

This package is the forecasting + maintain pillar of the portfolio:
leakage-free walk-forward evaluation against a persistence baseline, plus a
documented drift-detection and retraining loop.

Stage 1: data layer (offline fixture + cached yfinance fetch), return/lag/rolling
features with a no-lookahead contract, a persistence baseline and a linear
primary model running through one expanding-origin walk-forward harness. No
drift detection or retraining is implemented yet.
"""

__version__ = "0.1.0"
