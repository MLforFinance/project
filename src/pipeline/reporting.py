from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analytics import compute_drawdown, scale_to_target_vol


def _cost_label(transaction_cost_bps: float | None) -> str:
    if transaction_cost_bps is None:
        return ""
    return f" | TC = {float(transaction_cost_bps):g} bps"


def flatten_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for strategy, frame in panel.items():
        renamed = frame.copy()
        renamed.columns = [f"{strategy}__{column}" for column in renamed.columns]
        parts.append(renamed)
    if not parts:
        return pd.DataFrame()
    output = pd.concat(parts, axis=1).sort_index()
    output.index.name = "date"
    return output


def select_plot_columns(metrics_table: pd.DataFrame, portfolio_returns: pd.DataFrame) -> list[str]:
    selected = [column for column in ["SPY_buy_and_hold", "equal_weight_benchmark"] if column in portfolio_returns.columns]
    treatment = metrics_table[metrics_table["control_group"] == "treatment"]
    for family in ["naive", "black_litterman", "ridge"]:
        family_rows = treatment[treatment["family"] == family]
        if not family_rows.empty:
            best_strategy = family_rows.sort_values("sharpe_annualized", ascending=False).iloc[0]["strategy"]
            selected.append(str(best_strategy))
    return [column for column in dict.fromkeys(selected) if column in portfolio_returns.columns]


def _family_columns(portfolio_returns: pd.DataFrame, family_prefix: str) -> list[str]:
    return [column for column in portfolio_returns.columns if column.startswith(f"{family_prefix}_")]


def _base_strategy_name(strategy: str) -> str:
    return strategy.split("__", 1)[0]


def plot_cumulative_returns(
    returns_df: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    cumulative = (1.0 + returns_df).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in cumulative.columns:
        ax.plot(cumulative.index, cumulative[column], label=column)
    ax.set_title(f"Strategy vs Benchmark Cumulative Returns (Net of Transaction Costs){_cost_label(transaction_cost_bps)}")
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_drawdowns(
    returns_df: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    drawdowns = returns_df.apply(compute_drawdown)
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in drawdowns.columns:
        ax.plot(drawdowns.index, drawdowns[column], label=column)
    ax.set_title(f"Drawdowns (Net of Transaction Costs){_cost_label(transaction_cost_bps)}")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_scaled_log_returns(
    returns_df: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    scaled = scale_to_target_vol(returns_df)
    cumulative_log = np.log1p(scaled).cumsum()
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in cumulative_log.columns:
        ax.plot(cumulative_log.index, cumulative_log[column], label=column)
    ax.set_title(f"Cumulative Log Returns at 10% Volatility Scaling (Net){_cost_label(transaction_cost_bps)}")
    ax.set_ylabel("Cumulative log return")
    ax.set_xlabel("Date")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_all_methods_scaled_log_returns(
    portfolio_returns: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    scaled = scale_to_target_vol(portfolio_returns)
    cumulative_log = np.log1p(scaled).cumsum()

    family_styles = {
        "ridge": ("yellow", 1.2, 0.95),
        "naive": ("red", 1.2, 0.35),
        "mvo": ("blue", 1.2, 0.35),
        "black_litterman": ("green", 1.2, 0.35),
    }
    benchmark_styles = {
        "equal_weight_benchmark": ("magenta", 1.8, 1.0, "EW"),
        "SPY_buy_and_hold": ("cyan", 1.8, 1.0, "SPY"),
    }

    fig, ax = plt.subplots(figsize=(13, 7))
    for family, (color, linewidth, alpha) in family_styles.items():
        columns = _family_columns(cumulative_log, family)
        for idx, column in enumerate(columns):
            label = family.replace("_", "-").title() if idx == 0 else None
            ax.plot(cumulative_log.index, cumulative_log[column], color=color, linewidth=linewidth, alpha=alpha, label=label)

    for column, (color, linewidth, alpha, label) in benchmark_styles.items():
        if column in cumulative_log.columns:
            ax.plot(cumulative_log.index, cumulative_log[column], color=color, linewidth=linewidth, alpha=alpha, label=label)

    ax.set_title(f"Portfolio Cumulative Log Returns (Volatility Scaling, Net): vol_target = 10.0%{_cost_label(transaction_cost_bps)}")
    ax.set_ylabel("Cumulative Log Returns (Vol. Target of 10%)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_family_vs_benchmarks(
    portfolio_returns: pd.DataFrame,
    family: str,
    output_path: Path,
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    scaled = scale_to_target_vol(portfolio_returns)
    cumulative_log = np.log1p(scaled).cumsum()

    family_colors = {
        "naive": "blue",
        "mvo": "red",
        "black_litterman": "orchid",
        "ridge": "gold",
    }
    benchmark_styles = {
        "equal_weight_benchmark": ("green", "EW"),
        "SPY_buy_and_hold": ("black", "SPY"),
    }

    fig, ax = plt.subplots(figsize=(13, 7))
    for column, (color, label) in benchmark_styles.items():
        if column in cumulative_log.columns:
            ax.plot(cumulative_log.index, cumulative_log[column], color=color, linewidth=1.8, label=label)

    family_columns = _family_columns(cumulative_log, family)
    family_label = family.replace("_", "-").title()
    family_color = family_colors[family]
    for idx, column in enumerate(family_columns):
        ax.plot(
            cumulative_log.index,
            cumulative_log[column],
            color=family_color,
            linewidth=1.4,
            alpha=0.35 if idx > 0 else 0.55,
            label=family_label if idx == 0 else None,
        )

    if family != "mvo":
        for idx, column in enumerate(_family_columns(cumulative_log, "mvo")):
            ax.plot(
                cumulative_log.index,
                cumulative_log[column],
                color="red",
                linewidth=1.2,
                alpha=0.25,
                label="MVO" if idx == 0 else None,
            )

    ax.set_title(f"{family_label} Portfolio ({family}) vs Benchmarks (Net){_cost_label(transaction_cost_bps)}")
    ax.set_ylabel("Cumulative Log Returns (Vol. Target of 10%)")
    ax.set_xlabel("Date")
    ax.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_rolling_sharpe(
    returns: pd.Series,
    output_path: Path,
    window: int = 12,
    annualized: bool = False,
    show: bool = False,
    strategy_name: str | None = None,
    transaction_cost_bps: float | None = None,
) -> None:
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std().replace(0, np.nan)
    rolling_sharpe = rolling_mean / rolling_std
    if annualized:
        rolling_sharpe = rolling_sharpe * np.sqrt(12.0)

    title_suffix = "annualized" if annualized else "monthly"
    label_suffix = "annualized" if annualized else "monthly"
    display_name = strategy_name or returns.name or "strategy"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rolling_sharpe.index, rolling_sharpe, label=f"{display_name} | {window}-month rolling Sharpe ({label_suffix})")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Rolling Sharpe ({title_suffix}, net): {display_name}{_cost_label(transaction_cost_bps)}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_control_vs_treatment_boxplots(
    metrics_table: pd.DataFrame,
    output_path: Path,
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    comparisons = [
        ("naive_vs_random", "Naive", "Random"),
        ("bl_vs_mvo", "Black-Litterman", "MVO"),
        ("ridge_vs_random", "Ridge", "Random"),
    ]
    metrics = [
        ("sharpe_annualized", "Sharpe (Ann.)"),
        ("sortino_annualized", "Sortino (Ann.)"),
        ("max_drawdown", "MaxDD"),
        ("positive_return_pct", "% Positive"),
    ]

    fig, axes = plt.subplots(len(comparisons), len(metrics), figsize=(18, 10))
    for row_idx, (comparison, treatment_label, control_label) in enumerate(comparisons):
        subset = metrics_table[metrics_table["comparison"] == comparison]
        treatment = subset[subset["control_group"] == "treatment"]
        control = subset[subset["control_group"] == "control"]
        for col_idx, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            data = [control[metric_key].dropna().to_numpy(), treatment[metric_key].dropna().to_numpy()]
            ax.boxplot(data, tick_labels=[control_label, treatment_label])
            ax.set_title(f"{comparison}: {metric_label}{_cost_label(transaction_cost_bps)}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_forecast_mode_comparison(
    metrics_table: pd.DataFrame,
    output_path: Path,
    metric_key: str = "sharpe_annualized",
    show: bool = False,
    transaction_cost_bps: float | None = None,
) -> None:
    subset = metrics_table[metrics_table["forecast_mode"].isin(["hard", "soft"])].copy()
    if subset.empty:
        return

    subset["base_strategy"] = subset["strategy"].map(_base_strategy_name)
    pivot = subset.pivot_table(index="base_strategy", columns="forecast_mode", values=metric_key, aggfunc="first")
    pivot = pivot.dropna(subset=["hard", "soft"])
    if pivot.empty:
        return

    pivot = pivot.sort_values("soft", ascending=True)
    y = np.arange(len(pivot))
    height = 0.38

    fig_height = max(6, 0.28 * len(pivot) + 1.5)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.barh(y - height / 2, pivot["hard"], height=height, label="Hard", color="#c44e52", alpha=0.85)
    ax.barh(y + height / 2, pivot["soft"], height=height, label="Soft", color="#4c72b0", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel(metric_key.replace("_", " ").title())
    ax.set_title(f"Hard vs Soft Forecast Comparison ({metric_key.replace('_', ' ').title()}){_cost_label(transaction_cost_bps)}")
    ax.axvline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_result_tables(metrics_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "naive_result_table": metrics_table[metrics_table["family"].isin(["naive", "naive_random"])].copy(),
        "bl_mvo_result_table": metrics_table[metrics_table["family"].isin(["black_litterman", "mvo"])].copy(),
        "ridge_result_table": metrics_table[metrics_table["family"].isin(["ridge", "ridge_random"])].copy(),
    }
