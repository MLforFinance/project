from __future__ import annotations

import pandas as pd

from .config import DEFAULT_ETF_TICKERS


def ensure_month_start_index(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    obj = df.copy()
    obj.index = pd.to_datetime(obj.index).to_period("M").to_timestamp(how="start")
    obj = obj.sort_index()
    obj.index.name = getattr(df.index, "name", None) or "date"
    return obj


def prices_to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="all")
    returns = ensure_month_start_index(returns)
    return returns.dropna(how="any")


def download_etf_prices(
    tickers: list[str] | tuple[str, ...] = DEFAULT_ETF_TICKERS,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("yfinance is required for ETF downloads. Install it before running the backtest.") from exc

    raw = yf.download(
        tickers=list(tickers),
        start=None if start is None else pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
        interval="1mo",
        auto_adjust=False,
        progress=False,
    )
    if raw.empty:
        raise ValueError("Yahoo Finance returned no ETF data.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise ValueError("Could not find Adjusted Close or Close in Yahoo Finance response.")
    else:
        prices = raw.to_frame(name=tickers[0])

    prices = ensure_month_start_index(prices)
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.dropna(how="all")


def align_macro_and_returns(reduced_df: pd.DataFrame, returns_df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Index]:
    X_full = ensure_month_start_index(reduced_df)
    Y_full = ensure_month_start_index(returns_df)
    common_dates = X_full.index.intersection(Y_full.index).sort_values()
    X_common = X_full.loc[common_dates]
    Y_common = Y_full.loc[common_dates]

    Y_target = Y_common.shift(-1)
    valid_mask = ~Y_target.isna().any(axis=1)
    X_aligned = X_common.loc[valid_mask]
    Y_aligned = Y_target.loc[valid_mask]
    target_dates = X_aligned.index + pd.offsets.MonthBegin(1)

    return {
        "X": X_aligned,
        "Y": Y_aligned,
        "common_dates": common_dates,
        "target_dates": target_dates,
    }
