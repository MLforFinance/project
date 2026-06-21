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


def compute_performance_metrics(
    returns: pd.Series,
    gross_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    transaction_costs: pd.Series | None = None,
) -> dict[str, float]:
    gross = returns if gross_returns is None else gross_returns.reindex(returns.index).fillna(0.0)
    turnover_series = pd.Series(0.0, index=returns.index) if turnover is None else turnover.reindex(returns.index).fillna(0.0)
    cost_series = pd.Series(0.0, index=returns.index) if transaction_costs is None else transaction_costs.reindex(returns.index).fillna(0.0)

    return {
        "mean_return": float(returns.mean()),
        "gross_mean_return": float(gross.mean()),
        "volatility": float(returns.std()),
        "sharpe_monthly": sharpe_ratio(returns, annualize=False),
        "sharpe_annualized": sharpe_ratio(returns, annualize=True),
        "sortino_monthly": sortino_ratio(returns, annualize=False),
        "sortino_annualized": sortino_ratio(returns, annualize=True),
        "avg_drawdown": average_drawdown(returns),
        "max_drawdown": max_drawdown(returns),
        "positive_return_pct": positive_return_pct(returns),
        "cumulative_return": float((1.0 + returns).prod() - 1.0),
        "avg_monthly_turnover": float(turnover_series.mean()),
        "annualized_turnover": float(turnover_series.mean() * 12.0),
        "mean_transaction_cost": float(cost_series.mean()),
        "total_transaction_cost": float(cost_series.sum()),
        "cost_drag_mean_return": float(gross.mean() - returns.mean()),
    }


def scale_to_target_vol(returns: pd.DataFrame, target_annual_vol: float = TARGET_ANNUAL_VOL) -> pd.DataFrame:
    target_monthly_vol = target_annual_vol / np.sqrt(12.0)
    scaled = returns.copy()
    for column in scaled.columns:
        vol = scaled[column].std()
        if pd.notna(vol) and vol > 0:
            scaled[column] = scaled[column] * (target_monthly_vol / vol)
    return scaled


def strategy_meta(strategy_name: str, default_forecast_mode: str | None = None) -> dict[str, str | int]:
    base_name, separator, explicit_forecast_mode = strategy_name.partition("__")
    family, mode, l_token = base_name.rsplit("_", 2)
    l_value = int(l_token[1:])
    info = MODEL_FAMILIES[family]
    forecast_mode = explicit_forecast_mode if separator else (default_forecast_mode or "unspecified")
    return {
        "strategy": strategy_name,
        "family": family,
        "mode": mode,
        "l_value": l_value,
        "forecast_mode": forecast_mode,
        "control_group": info["control_group"],
        "comparison": info["comparison"],
    }


def is_strategy_column(column: str) -> bool:
    base_name = column.split("__", 1)[0]
    return any(base_name.startswith(family + "_") for family in MODEL_FAMILIES)


def build_metrics_table(
    portfolio_returns: pd.DataFrame,
    gross_returns: pd.DataFrame | None = None,
    turnover: pd.DataFrame | None = None,
    transaction_costs: pd.DataFrame | None = None,
    default_forecast_mode: str | None = None,
) -> pd.DataFrame:
    rows = []
    for column in portfolio_returns.columns:
        if is_strategy_column(column):
            meta = strategy_meta(column, default_forecast_mode=default_forecast_mode)
        else:
            meta = {
                "strategy": column,
                "family": column,
                "mode": "benchmark",
                "l_value": 0,
                "forecast_mode": "benchmark",
                "control_group": "benchmark",
                "comparison": "benchmark",
            }
        rows.append(
            {
                **meta,
                **compute_performance_metrics(
                    portfolio_returns[column],
                    gross_returns=None if gross_returns is None or column not in gross_returns.columns else gross_returns[column],
                    turnover=None if turnover is None or column not in turnover.columns else turnover[column],
                    transaction_costs=None if transaction_costs is None or column not in transaction_costs.columns else transaction_costs[column],
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["comparison", "family", "forecast_mode", "mode", "l_value", "strategy"]).reset_index(drop=True)
