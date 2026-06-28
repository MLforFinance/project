"""Run the focused negative-month S&P 500 risk model.

Run with:

    export FRED_API_KEY="your_key_here"
    python -m macro_prediction.main

Only this model is supported here: a walk-forward logistic regression that
predicts whether next month's S&P 500 return will be negative, then converts
that probability into an S&P/T-bill allocation. Edit ``CONFIG`` below to change
thresholds, costs, dates, and output location.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fredapi import Fred

from macro_prediction.downside import (
    fetch_sp500_monthly_returns,
    save_bad_month_run,
    walk_forward_bad_month_predictions,
)
from macro_prediction.fred import FredFeatureBuilder, default_macro_series, to_monthly
from macro_prediction.strategy import make_weighted_returns, plot_equity_and_drawdown, summarize_strategy_by_period


@dataclass(frozen=True)
class ModelConfig:
    # Data
    observation_start: str = "1985-01-01"
    observation_end: str | None = None
    vintage_date: str | None = None
    output_dir: Path = Path("artifacts/negative_month_model")

    # Walk-forward model
    mode: str = "expanding"
    min_train_months: int = 120
    rolling_window_months: int = 120
    target_return_threshold: float = 0.0
    include_market_features: bool = False

    # Allocation rule. Bands with 33/67 probability thresholds.
    allocation_rule: str = "bands"
    lower_probability_threshold: float = 0.33
    upper_probability_threshold: float = 0.67
    mid_equity_weight: float = 0.50
    binary_probability_threshold: float = 0.67
    cost_per_unit_turnover: float = 0.0005

    # Evaluation periods
    tune_start: str = "1995-01-01"
    tune_end: str = "2014-12-31"
    test_start: str = "2015-01-01"
    test_end: str = "2026-12-31"


CONFIG = ModelConfig()


def main(config: ModelConfig = CONFIG) -> None:
    """Run the configured model and write artifacts."""

    config.output_dir.mkdir(parents=True, exist_ok=True)

    series = default_macro_series()
    fred_builder = FredFeatureBuilder(series)
    raw_macro = fred_builder.fetch_raw(
        observation_start=config.observation_start,
        observation_end=config.observation_end,
        vintage_date=config.vintage_date,
    )
    raw_macro.to_csv(config.output_dir / "raw_macro_data.csv")

    sp500_returns = fetch_sp500_monthly_returns(start="1960-01-01")
    sp500_returns.to_csv(config.output_dir / "sp500_monthly_returns.csv")

    cash_returns = fetch_tbill_cash_returns(start="1960-01-01")
    cash_returns.to_csv(config.output_dir / "tbill_cash_returns.csv")

    run = walk_forward_bad_month_predictions(
        raw_macro,
        series,
        sp500_returns,
        mode=config.mode,
        bad_return_threshold=config.target_return_threshold,
        min_train_months=config.min_train_months,
        window_months=config.rolling_window_months if config.mode == "rolling_10y" else None,
        probability_threshold=config.binary_probability_threshold,
        include_market_features=config.include_market_features,
    )
    save_bad_month_run(run, config.output_dir)

    strategy = make_weighted_returns(
        run.predictions.rename(
            columns={"bad_month_probability": "risk_probability"}),
        probability_column="risk_probability",
        cash_returns=cash_returns,
        rule=config.allocation_rule,
        probability_threshold=config.binary_probability_threshold,
        lower_threshold=config.lower_probability_threshold,
        upper_threshold=config.upper_probability_threshold,
        mid_weight=config.mid_equity_weight,
        cost_per_unit_turnover=config.cost_per_unit_turnover,
    )
    strategy.to_csv(config.output_dir / "strategy_returns.csv")

    periods = {
        "full": ("1900-01-01", "2100-01-01"),
        "tune": (config.tune_start, config.tune_end),
        "test": (config.test_start, config.test_end),
    }
    summary = summarize_strategy_by_period(strategy, periods)
    summary["exposure_pct"] = [
        strategy.loc[start:end, "position"].mean(
        ) * 100.0 if row_strategy != "buy_hold" else 100.0
        for period, row_strategy in summary[["period", "strategy"]].itertuples(index=False, name=None)
        for start, end in [periods[period]]
    ]
    summary.to_csv(config.output_dir / "strategy_summary.csv", index=False)

    plot_equity_and_drawdown(
        strategy,
        config.output_dir / "strategy_equity_drawdown.png",
        title="Negative-month probability sizing strategy",
    )

    print("Saved outputs to", config.output_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def fetch_tbill_cash_returns(start: str) -> pd.Series:
    """Fetch monthly 3-month T-bill cash return approximation from FRED TB3MS."""

    fred = Fred()
    tb3ms = fred.get_series("TB3MS", observation_start=start)
    tb3ms_monthly = to_monthly(tb3ms, frequency="monthly")
    cash_returns = tb3ms_monthly / 100.0 / 12.0
    cash_returns.name = "cash_return"
    return cash_returns


if __name__ == "__main__":
    main()
