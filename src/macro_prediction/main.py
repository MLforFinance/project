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
    random_bad_month_predictions,
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
    include_market_features: bool = True

    # Allocation rule. Bands with 33/67 probability thresholds.
    allocation_rule: str = "bands"
    lower_probability_threshold: float = 0.33
    upper_probability_threshold: float = 0.67
    mid_equity_weight: float = 0.50
    binary_probability_threshold: float = 0.67
    cost_per_unit_turnover: float = 0.0005

    # Random bad-month baseline
    random_baseline_enabled: bool = True
    random_baseline_seed: int = 42
    random_baseline_trials: int = 100
    random_bad_month_share: float | None = None

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

    if config.random_baseline_enabled:
        random_run = random_bad_month_predictions(
            raw_macro,
            sp500_returns,
            mode="random",
            bad_return_threshold=config.target_return_threshold,
            min_train_months=config.min_train_months,
            probability_threshold=config.binary_probability_threshold,
            seed=config.random_baseline_seed,
            bad_month_share=config.random_bad_month_share,
        )
        save_bad_month_run(random_run, config.output_dir)

        random_strategy = make_weighted_returns(
            random_run.predictions.rename(columns={"bad_month_probability": "risk_probability"}),
            probability_column="risk_probability",
            cash_returns=cash_returns,
            rule=config.allocation_rule,
            probability_threshold=config.binary_probability_threshold,
            lower_threshold=config.lower_probability_threshold,
            upper_threshold=config.upper_probability_threshold,
            mid_weight=config.mid_equity_weight,
            cost_per_unit_turnover=config.cost_per_unit_turnover,
        )
        random_strategy.to_csv(config.output_dir / "random_strategy_returns.csv")

        random_summary = summarize_strategy_by_period(random_strategy, periods)
        random_summary["exposure_pct"] = [
            random_strategy.loc[start:end, "position"].mean() * 100.0 if row_strategy != "buy_hold" else 100.0
            for period, row_strategy in random_summary[["period", "strategy"]].itertuples(index=False, name=None)
            for start, end in [periods[period]]
        ]
        random_summary.to_csv(config.output_dir / "random_strategy_summary.csv", index=False)

        plot_equity_and_drawdown(
            random_strategy,
            config.output_dir / "random_strategy_equity_drawdown.png",
            title="Random bad-month sizing baseline",
        )

        random_trials = run_random_baseline_trials(
            raw_macro,
            sp500_returns,
            cash_returns,
            periods,
            config=config,
        )
        random_trials.to_csv(config.output_dir / "random_strategy_trials.csv", index=False)
        random_trial_summary = summarize_random_trials(random_trials)
        random_trial_summary.to_csv(config.output_dir / "random_strategy_trial_summary.csv", index=False)

    print("Saved outputs to", config.output_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if config.random_baseline_enabled:
        print("\nRandom bad-month baseline:")
        print(random_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nRandom baseline trials ({config.random_baseline_trials} seeds):")
        print(random_trial_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def run_random_baseline_trials(
    raw_macro: pd.DataFrame,
    sp500_returns: pd.Series,
    cash_returns: pd.Series,
    periods: dict[str, tuple[str, str]],
    *,
    config: ModelConfig,
) -> pd.DataFrame:
    """Run many random bad-month baselines to show the noise band."""

    if config.random_baseline_trials < 1:
        raise ValueError("random_baseline_trials must be at least 1")

    rows = []
    for offset in range(config.random_baseline_trials):
        seed = config.random_baseline_seed + offset
        random_run = random_bad_month_predictions(
            raw_macro,
            sp500_returns,
            mode="random",
            bad_return_threshold=config.target_return_threshold,
            min_train_months=config.min_train_months,
            probability_threshold=config.binary_probability_threshold,
            seed=seed,
            bad_month_share=config.random_bad_month_share,
        )
        random_strategy = make_weighted_returns(
            random_run.predictions.rename(columns={"bad_month_probability": "risk_probability"}),
            probability_column="risk_probability",
            cash_returns=cash_returns,
            rule=config.allocation_rule,
            probability_threshold=config.binary_probability_threshold,
            lower_threshold=config.lower_probability_threshold,
            upper_threshold=config.upper_probability_threshold,
            mid_weight=config.mid_equity_weight,
            cost_per_unit_turnover=config.cost_per_unit_turnover,
        )
        trial_summary = summarize_strategy_by_period(random_strategy, periods)
        trial_summary = trial_summary[trial_summary["strategy"] == "cash_when_risky"].copy()
        trial_summary["seed"] = seed
        trial_summary["exposure_pct"] = [
            random_strategy.loc[start:end, "position"].mean() * 100.0
            for period in trial_summary["period"]
            for start, end in [periods[period]]
        ]
        rows.append(trial_summary)

    return pd.concat(rows, ignore_index=True)


def summarize_random_trials(random_trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate random baseline trial outcomes by period."""

    metrics = [
        "total_return_pct",
        "annual_return_pct",
        "annual_volatility_pct",
        "sharpe",
        "max_drawdown_pct",
        "exposure_pct",
    ]
    grouped = random_trials.groupby("period", sort=False)[metrics]
    summary = grouped.agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        column[0] if column[1] == "" else f"{column[0]}_{column[1]}"
        for column in summary.columns
    ]
    return summary


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
