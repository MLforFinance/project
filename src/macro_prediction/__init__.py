"""Macroeconomic prediction experiments."""

from macro_prediction.fred import FredFeatureBuilder, FredSeriesSpec, default_fredmd_representative_series
from macro_prediction.hmm import GaussianHMMRegimeModel

__all__ = [
    "FredFeatureBuilder",
    "FredSeriesSpec",
    "GaussianHMMRegimeModel",
    "default_fredmd_representative_series",
    "__version__",
]

__version__ = "0.1.0"
