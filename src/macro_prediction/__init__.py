"""Negative-month S&P 500 risk model."""

from macro_prediction.fred import FredFeatureBuilder, FredSeriesSpec, default_macro_series

__all__ = [
    "FredFeatureBuilder",
    "FredSeriesSpec",
    "default_macro_series",
    "__version__",
]

__version__ = "0.1.0"
