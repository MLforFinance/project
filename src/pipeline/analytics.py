from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODEL_FAMILIES, TARGET_ANNUAL_VOL


def compute_drawdown(returns: pd.Series) -> pd.Series:
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    return cumulative / peak - 1.0


def sharpe_ratio(returns: pd.Series, annualize: bool = False) -> float:
    std = returns.std()
    if pd.isna(std) or std == 0:
        return 0.0
    value = float(returns.mean() / std)
    return float(value * np.sqrt(12.0)) if annualize else value


def sortino_ratio(returns: pd.Series, annualize: bool = False) -> float:
    target = 0.0
    excess = returns - target
    # clip, don't filter
    downside = np.minimum(0, excess)
    downside_std = np.sqrt((downside ** 2).mean())
    if pd.isna(downside_std) or downside_std == 0:
        return 0.0

    value = float(excess.mean() / downside_std)
    return float(value * np.sqrt(12.0)) if annualize else value


def average_drawdown(returns: pd.Series) -> float:
    drawdown = compute_drawdown(returns)
    negative = drawdown[drawdown < 0]
    if negative.empty:
        return 0.0
    return float(negative.mean())


def max_drawdown(returns: pd.Series) -> float:
    drawdown = compute_drawdown(returns)
    if drawdown.empty:
        return 0.0
    return float(drawdown.min())


def positive_return_pct(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def compute_performance_metrics(returns: pd.Series) -> dict[str, float]:
    return {
        "mean_return": float(returns.mean()),
        "volatility": float(returns.std()),
        "sharpe_monthly": sharpe_ratio(returns, annualize=False),
        "sharpe_annualized": sharpe_ratio(returns, annualize=True),
        "sortino_monthly": sortino_ratio(returns, annualize=False),
        "sortino_annualized": sortino_ratio(returns, annualize=True),
        "avg_drawdown": average_drawdown(returns),
        "max_drawdown": max_drawdown(returns),
        "positive_return_pct": positive_return_pct(returns),
        "cumulative_return": float((1.0 + returns).prod() - 1.0),
    }


def scale_to_target_vol(returns: pd.DataFrame, target_annual_vol: float = TARGET_ANNUAL_VOL) -> pd.DataFrame:
    target_monthly_vol = target_annual_vol / np.sqrt(12.0)
    scaled = returns.copy()
    for column in scaled.columns:
        vol = scaled[column].std()
        if pd.notna(vol) and vol > 0:
            scaled[column] = scaled[column] * (target_monthly_vol / vol)
    return scaled


def strategy_meta(strategy_name: str) -> dict[str, str | int]:
    family, mode, l_token = strategy_name.rsplit("_", 2)
    l_value = int(l_token[1:])
    info = MODEL_FAMILIES[family]
    return {
        "strategy": strategy_name,
        "family": family,
        "mode": mode,
        "l_value": l_value,
        "control_group": info["control_group"],
        "comparison": info["comparison"],
    }


def is_strategy_column(column: str) -> bool:
    return any(column.startswith(family + "_") for family in MODEL_FAMILIES)


def build_metrics_table(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in portfolio_returns.columns:
        if is_strategy_column(column):
            meta = strategy_meta(column)
        else:
            meta = {
                "strategy": column,
                "family": column,
                "mode": "benchmark",
                "l_value": 0,
                "control_group": "benchmark",
                "comparison": "benchmark",
            }
        rows.append(
            {**meta, **compute_performance_metrics(portfolio_returns[column])})
    return pd.DataFrame(rows).sort_values(["comparison", "family", "mode", "l_value", "strategy"]).reset_index(drop=True)
