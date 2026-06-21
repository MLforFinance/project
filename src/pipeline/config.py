from __future__ import annotations

DEFAULT_CLUSTER_COUNT = 5
DEFAULT_TARGET_VARIANCE = 0.95

DEFAULT_KERNEL = "poly"
DEFAULT_KERNEL_COMPONENTS = 6
DEFAULT_GAMMA = None
DEFAULT_DEGREE = 2
DEFAULT_COEF0 = 1.0
DEFAULT_RANDOM_STATE = 42

DEFAULT_PLOT_FORMAT = "svg"
DEFAULT_TRIM_ROWS = None
DEFAULT_ETF_TICKERS = ["SPY", "XLB", "XLE", "XLF",
                       "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEFAULT_WINDOW_SIZE = 48
DEFAULT_L_VALUES = (2, 3, 4)
DEFAULT_SIZING_MODES = ("lo", "lns", "los", "mx")
DEFAULT_TRANSACTION_COST_BPS = 20.0
DEFAULT_FORECAST_MODE = "soft"
FORECAST_MODE_CHOICES = ("hard", "soft", "both")
TARGET_ANNUAL_VOL = 0.10
BLACK_LITTERMAN_TAU = 0.05
RANDOM_SEED = 42

MODEL_FAMILIES = {
    "naive": {"kind": "naive", "control_group": "treatment", "comparison": "naive_vs_random"},
    "naive_random": {"kind": "naive", "control_group": "control", "comparison": "naive_vs_random", "random_regimes": True},
    "black_litterman": {"kind": "black_litterman", "control_group": "treatment", "comparison": "bl_vs_mvo"},
    "mvo": {"kind": "mvo", "control_group": "control", "comparison": "bl_vs_mvo"},
    "ridge": {"kind": "ridge", "control_group": "treatment", "comparison": "ridge_vs_random"},
    "ridge_random": {"kind": "ridge", "control_group": "control", "comparison": "ridge_vs_random", "random_regimes": True},
}
