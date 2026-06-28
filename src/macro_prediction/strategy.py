"""Strategy evaluation helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_cash_when_risky_returns(
    predictions: pd.DataFrame,
    *,
    probability_threshold: float,
    cost_per_switch: float = 0.0,
) -> pd.DataFrame:
    """Create monthly strategy returns from bad-month probabilities.

    The strategy is long S&P 500 when bad-month probability is below the
    threshold and in cash otherwise. Transaction cost is charged whenever the
    position changes from the prior month.
    """

    data = predictions.dropna(subset=["bad_month_probability", "actual_sp500_return"]).copy()
    data["position"] = (data["bad_month_probability"] < probability_threshold).astype(float)
    data["switch"] = data["position"].diff().abs().fillna(0.0)
    data["strategy_return"] = data["position"] * data["actual_sp500_return"]
    if cost_per_switch:
        data["strategy_return"] = data["strategy_return"] - data["switch"] * cost_per_switch
    data["buy_hold_return"] = data["actual_sp500_return"]
    return data


def make_weighted_returns(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    equity_returns: pd.Series | None = None,
    cash_returns: pd.Series | None = None,
    rule: str = "linear",
    probability_threshold: float = 0.67,
    lower_threshold: float = 0.33,
    upper_threshold: float = 0.67,
    mid_weight: float = 0.50,
    cost_per_unit_turnover: float = 0.0,
) -> pd.DataFrame:
    """Create strategy returns from probability-based equity weights.

    Rules:
    - binary: 100% equity below probability_threshold, else cash.
    - linear: equity weight is 1 - probability.
    - bands: 100% below lower_threshold, mid_weight between thresholds, cash above upper_threshold.
    """

    data = predictions.copy()
    if equity_returns is not None:
        data = data.join(equity_returns.rename("actual_sp500_return"), how="left")
    data = data.dropna(subset=[probability_column, "actual_sp500_return"]).copy()

    probability = data[probability_column].clip(0.0, 1.0)
    if rule == "binary":
        data["position"] = (probability < probability_threshold).astype(float)
    elif rule == "linear":
        data["position"] = 1.0 - probability
    elif rule == "bands":
        data["position"] = 1.0
        data.loc[probability >= lower_threshold, "position"] = mid_weight
        data.loc[probability >= upper_threshold, "position"] = 0.0
    else:
        raise ValueError("rule must be 'binary', 'linear', or 'bands'")

    if cash_returns is None:
        cash = pd.Series(0.0, index=data.index, name="cash_return")
    else:
        cash = cash_returns.reindex(data.index).fillna(0.0).rename("cash_return")
    data["cash_return"] = cash
    data["turnover"] = data["position"].diff().abs().fillna(0.0)
    data["strategy_return"] = (
        data["position"] * data["actual_sp500_return"]
        + (1.0 - data["position"]) * data["cash_return"]
        - data["turnover"] * cost_per_unit_turnover
    )
    data["buy_hold_return"] = data["actual_sp500_return"]
    return data


def performance_summary(returns: pd.Series) -> pd.Series:
    """Calculate simple monthly-strategy performance metrics."""

    r = returns.dropna()
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    ann_return = equity.iloc[-1] ** (12.0 / len(r)) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(12.0)
    return pd.Series(
        {
            "months": len(r),
            "total_return_pct": (equity.iloc[-1] - 1.0) * 100.0,
            "annual_return_pct": ann_return * 100.0,
            "annual_volatility_pct": ann_vol * 100.0,
            "sharpe": ann_return / ann_vol if ann_vol else np.nan,
            "max_drawdown_pct": drawdown.min() * 100.0,
            "avg_monthly_return_pct": r.mean() * 100.0,
            "monthly_volatility_pct": r.std(ddof=1) * 100.0,
        }
    )


def summarize_strategy_by_period(
    data: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """Summarize strategy and buy-hold performance over named periods."""

    rows = []
    for period, (start, end) in periods.items():
        subset = data.loc[start:end]
        if subset.empty:
            continue
        strategy = performance_summary(subset["strategy_return"])
        buy_hold = performance_summary(subset["buy_hold_return"])
        rows.append({"period": period, "strategy": "cash_when_risky", **strategy.to_dict()})
        rows.append({"period": period, "strategy": "buy_hold", **buy_hold.to_dict()})
    return pd.DataFrame(rows)


def plot_equity_and_drawdown(
    data: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot strategy vs buy-hold equity and drawdown."""

    strategy_equity = (1.0 + data["strategy_return"]).cumprod()
    buy_hold_equity = (1.0 + data["buy_hold_return"]).cumprod()
    strategy_dd = strategy_equity / strategy_equity.cummax() - 1.0
    buy_hold_dd = buy_hold_equity / buy_hold_equity.cummax() - 1.0

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[2, 1])
    axes[0].plot(strategy_equity.index, strategy_equity, label="Strategy", linewidth=1.8)
    axes[0].plot(buy_hold_equity.index, buy_hold_equity, label="Buy & hold", linewidth=1.4, alpha=0.85)
    risk_off = data[data["position"] == 0.0]
    if not risk_off.empty:
        axes[0].scatter(risk_off.index, strategy_equity.reindex(risk_off.index), s=8, color="#d62728", label="Risk off")
    axes[0].set_title(title)
    axes[0].set_ylabel("Growth of $1")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    axes[1].plot(strategy_dd.index, strategy_dd * 100.0, label="Strategy", linewidth=1.6)
    axes[1].plot(buy_hold_dd.index, buy_hold_dd * 100.0, label="Buy & hold", linewidth=1.2, alpha=0.85)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
