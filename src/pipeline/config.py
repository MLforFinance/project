from __future__ import annotations

DEFAULT_CLUSTER_COUNT = 5
DEFAULT_TARGET_VARIANCE = 0.95
DEFAULT_PLOT_FORMAT = "svg"
DEFAULT_TRIM_ROWS = None
DEFAULT_ETF_TICKERS = ["SPY", "XLB", "XLE", "XLF",
                       "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
DEFAULT_WINDOW_SIZE = 48
DEFAULT_L_VALUES = (2, 3, 4)
DEFAULT_SIZING_MODES = ("lo", "lns", "los", "mx")
DEFAULT_TRANSACTION_COST_BPS = 0
DEFAULT_FORECAST_MODE = "soft"
FORECAST_MODE_CHOICES = ("hard", "soft", "both")
DEFAULT_REGIME_MODEL = "kmeans"
REGIME_MODEL_CHOICES = ("kmeans", "isolation_umap_hmm")


DEFAULT_PROB_MODE = "soft"
PROB_MODE_CHOICES = ("hard", "soft") 
DEFAULT_UMAP_COMPONENTS = 4
DEFAULT_ISO_SCORE_SCALE = 5.0
CONTAMINATION_RATE = 0.08


DEFAULT_EMA_SPAN = 12

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
