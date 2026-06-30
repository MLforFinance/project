from __future__ import annotations
import numpy as np
import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from ..data_processing.kernel_PCA import optimal_kernel_PCA
    from ..data_processing.preprocessing import preprocess_fred_md
    from ..models.Isolation import Isolation_Euclidean_KMeans
except ImportError:  # pragma: no cover
    from data_processing.kernel_PCA import optimal_kernel_PCA
    from data_processing.preprocessing import preprocess_fred_md
    from models.Isolation import Isolation_Euclidean_KMeans




from .backtest import run_walk_forward_backtest
from .config import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_ETF_TICKERS,
    DEFAULT_FORECAST_MODE,
    FORECAST_MODE_CHOICES,
    DEFAULT_PLOT_FORMAT,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_CASH_TICKER,
    DEFAULT_ENABLE_CASH_ASSET,
    DEFAULT_ENABLE_DYNAMIC_RISK_OVERLAY,
    DEFAULT_FIXED_OVERLAY_EXPOSURE,
    DEFAULT_OVERLAY_HARD_DRAWDOWN,
    DEFAULT_OVERLAY_HARD_EXPOSURE,
    DEFAULT_OVERLAY_GOOD_PROBABILITY_THRESHOLD,
    DEFAULT_OVERLAY_GOOD_REGIME_COUNT,
    DEFAULT_OVERLAY_LOOKBACK_MONTHS,
    DEFAULT_OVERLAY_SOFT_DRAWDOWN,
    DEFAULT_OVERLAY_SOFT_EXPOSURE,
    DEFAULT_TRIM_ROWS,    
    DEFAULT_OUTLIER_METHOD,
    DEFAULT_IMPUTATION_METHOD,
    DEFAULT_IMPUTATION_BURN_IN,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_KERNEL,
    DEFAULT_KERNEL_COMPONENTS,
    DEFAULT_GAMMA,
    DEFAULT_DEGREE,
    DEFAULT_COEF0,
    DEFAULT_RANDOM_STATE,
)
from .data_sources import align_macro_and_returns, download_etf_prices, ensure_month_start_index, prices_to_returns
from .paths import default_data_dir, derive_backtest_paths, derive_output_paths, derive_plot_paths, discover_input_csv
from .regime_pipeline import compute_transition_matrix, next_regime_probs, plot_kmeans_regimes, plot_pca_clusters, renormalize_probabilities
from .reporting import build_result_tables, plot_all_methods_scaled_log_returns, plot_control_vs_treatment_boxplots, plot_cumulative_returns, plot_drawdowns, plot_family_vs_benchmarks, plot_forecast_mode_comparison, plot_rolling_sharpe, plot_scaled_log_returns, select_plot_columns

def run_pipeline(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    kernel: str = DEFAULT_KERNEL,
    kernel_components: int = DEFAULT_KERNEL_COMPONENTS,
    gamma: float | None = DEFAULT_GAMMA,
    degree: int = DEFAULT_DEGREE,
    coef0: float = DEFAULT_COEF0,
    regime_count: int = DEFAULT_CLUSTER_COUNT,
    trim_rows: int | None = DEFAULT_TRIM_ROWS,
    outlier_method: str = DEFAULT_OUTLIER_METHOD,
    imputation_method: str = DEFAULT_IMPUTATION_METHOD,
    imputation_burn_in: int = DEFAULT_IMPUTATION_BURN_IN,
    plot: bool = True,
    plot_format: str = DEFAULT_PLOT_FORMAT,
    show_plots: bool = False,
    ridge_alpha: float = 1.0,
    etf_tickers: list[str] | tuple[str, ...] = DEFAULT_ETF_TICKERS,
    download_etfs: bool = False,
    backtest: bool = False,
    window_size: int = DEFAULT_WINDOW_SIZE,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    forecast_mode: str = DEFAULT_FORECAST_MODE,
    enable_cash_asset: bool = DEFAULT_ENABLE_CASH_ASSET,
    fixed_overlay_exposure: float = DEFAULT_FIXED_OVERLAY_EXPOSURE,
    cash_ticker: str = DEFAULT_CASH_TICKER,
    enable_dynamic_risk_overlay: bool = DEFAULT_ENABLE_DYNAMIC_RISK_OVERLAY,
    overlay_lookback_months: int = DEFAULT_OVERLAY_LOOKBACK_MONTHS,
    overlay_soft_drawdown: float = DEFAULT_OVERLAY_SOFT_DRAWDOWN,
    overlay_hard_drawdown: float = DEFAULT_OVERLAY_HARD_DRAWDOWN,
    overlay_soft_exposure: float = DEFAULT_OVERLAY_SOFT_EXPOSURE,
    overlay_hard_exposure: float = DEFAULT_OVERLAY_HARD_EXPOSURE,
    overlay_good_probability_threshold: float = DEFAULT_OVERLAY_GOOD_PROBABILITY_THRESHOLD,
    overlay_good_regime_count: int = DEFAULT_OVERLAY_GOOD_REGIME_COUNT,
    asset_returns_df: pd.DataFrame | None = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    data_dir = default_data_dir()
    input_path = Path(input_csv) if input_csv is not None else discover_input_csv(data_dir)
    output_dir_path = Path(output_dir) if output_dir is not None else input_path.parent
    output_dir_path.mkdir(parents=True, exist_ok=True)

    processed_path, reduced_path, regimes_path = derive_output_paths(input_path, output_dir_path)
    pca_plot_path, timeline_plot_path = derive_plot_paths(input_path, output_dir_path, plot_format)
    backtest_paths = derive_backtest_paths(input_path, output_dir_path)

    processed_df, preprocessing_info = preprocess_fred_md(
        input_path,
        processed_path,
        trim_rows=trim_rows,
        outlier_method=outlier_method,
        imputation_method=imputation_method,
        imputation_burn_in=imputation_burn_in,
    )
    processed_df = ensure_month_start_index(processed_df)

    reduced_df, n_components, pca_model = optimal_kernel_PCA(
        processed_df,
        kernel=kernel,
        n_components=kernel_components,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    )
    reduced_df = ensure_month_start_index(reduced_df)

    reduced_df.to_csv(reduced_path)

    full_regimes, pred_isolation, pred_kmeans, kmeans_centers, minority_cluster, full_probs = Isolation_Euclidean_KMeans(
        reduced_df,
        r=regime_count,
        random_state=random_state,
    )

    prob_columns = [f"regime_prob_{i}" for i in range(regime_count + 1)]

    regimes_series = pd.Series(full_regimes, index=reduced_df.index, name="regime")
    probs_df = pd.DataFrame(full_probs, index=reduced_df.index, columns=prob_columns)
    pd.concat([reduced_df, regimes_series, probs_df], axis=1).to_csv(regimes_path)

    transition_matrix = compute_transition_matrix(probs_df, regime_count + 1)
    current_regime_probs = pd.Series(renormalize_probabilities(probs_df.iloc[-1].to_numpy()), index=prob_columns, name="current_regime_prob")
    next_regime_prob_series = pd.Series(next_regime_probs(current_regime_probs.to_numpy(), transition_matrix.to_numpy()), index=prob_columns, name="next_regime_prob")

    prices_df = None
    returns_df = None if asset_returns_df is None else ensure_month_start_index(asset_returns_df)
    aligned_data = None
    backtest_results = None
    result_tables = None

    if download_etfs:
        prices_df = download_etf_prices(etf_tickers, start=reduced_df.index.min(), end=reduced_df.index.max() + pd.offsets.MonthBegin(2))
        returns_df = prices_to_returns(prices_df)
        prices_df.to_csv(backtest_paths["etf_prices"])
        returns_df.to_csv(backtest_paths["etf_returns"])

    if returns_df is not None:
        # For reporting and full-sample regime plots, reduced_df is still computed once above.
        # For the backtest, however, use the already preprocessed/imputed macro panel
        # and refit Kernel PCA inside each expanding window. This hybrid version
        # intentionally performs missing-value handling once, but avoids full-sample
        # leakage in the Kernel PCA feature extraction used by the walk-forward test.
        macro_for_backtest = processed_df if backtest else reduced_df
        aligned_data = align_macro_and_returns(macro_for_backtest, returns_df)

    if backtest:
        if aligned_data is None:
            raise ValueError("Backtest requested but ETF returns are unavailable. Use --download-etfs or pass asset_returns_df.")
        backtest_results = run_walk_forward_backtest(
            aligned_data["X"],
            aligned_data["Y"],
            aligned_data["target_dates"],
            regime_count=regime_count,
            window_size=window_size,
            ridge_alpha=ridge_alpha,
            transaction_cost_bps=transaction_cost_bps,
            forecast_mode=forecast_mode,
            enable_cash_asset=enable_cash_asset,
            fixed_overlay_exposure=fixed_overlay_exposure,
            cash_ticker=cash_ticker,
            enable_dynamic_risk_overlay=enable_dynamic_risk_overlay,
            overlay_lookback_months=overlay_lookback_months,
            overlay_soft_drawdown=overlay_soft_drawdown,
            overlay_hard_drawdown=overlay_hard_drawdown,
            overlay_soft_exposure=overlay_soft_exposure,
            overlay_hard_exposure=overlay_hard_exposure,
            overlay_good_probability_threshold=overlay_good_probability_threshold,
            overlay_good_regime_count=overlay_good_regime_count,
            rolling_kernel_pca=True,
            kernel=kernel,
            kernel_components=kernel_components,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
        )
        backtest_results["portfolio_returns"].to_csv(backtest_paths["portfolio_returns"])
        backtest_results["gross_portfolio_returns"].to_csv(backtest_paths["gross_portfolio_returns"])
        backtest_results["turnover"].to_csv(backtest_paths["turnover"])
        backtest_results["transaction_costs"].to_csv(backtest_paths["transaction_costs"])
        backtest_results["overlay_exposures"].to_csv(backtest_paths["overlay_exposures"])
        backtest_results["overlay_recent_drawdowns"].to_csv(backtest_paths["overlay_recent_drawdowns"])
        backtest_results["overlay_good_probabilities"].to_csv(backtest_paths["overlay_good_probabilities"])
        backtest_results["overlay_actions"].to_csv(backtest_paths["overlay_actions"])
        backtest_results["weights"].to_csv(backtest_paths["weights"])
        backtest_results["predictions"].to_csv(backtest_paths["predictions"])
        backtest_results["metrics_table"].to_csv(backtest_paths["metrics_table"], index=False)
        metrics_payload = {
            "transaction_cost_bps": backtest_results["transaction_cost_bps"],
            "preprocessing_scope": "full_sample_once",
            "kernel_pca_scope": "expanding_window_refit",
            "rolling_kernel_pca": backtest_results.get("rolling_kernel_pca", False),
            "kernel": backtest_results.get("kernel", kernel),
            "kernel_components": backtest_results.get("kernel_components", kernel_components),
            "forecast_mode": backtest_results["forecast_mode"],
            "forecast_modes_evaluated": backtest_results["forecast_modes_evaluated"],
            "cash_ticker": backtest_results.get("cash_ticker"),
            "fixed_overlay_exposure": backtest_results.get("fixed_overlay_exposure"),
            "dynamic_risk_overlay_enabled": backtest_results.get("dynamic_risk_overlay_enabled"),
            "dynamic_risk_overlay_rules": backtest_results.get("dynamic_risk_overlay_rules"),
            "strategies": backtest_results["metrics_json"],
        }
        with backtest_paths["metrics"].open("w", encoding="utf-8") as handle:
            json.dump(metrics_payload, handle, indent=2)

        result_tables = build_result_tables(backtest_results["metrics_table"])
        result_tables["naive_result_table"].to_csv(backtest_paths["naive_result_table"], index=False)
        result_tables["bl_mvo_result_table"].to_csv(backtest_paths["bl_mvo_result_table"], index=False)
        result_tables["ridge_result_table"].to_csv(backtest_paths["ridge_result_table"], index=False)

        if plot:
            selected_columns = select_plot_columns(backtest_results["metrics_table"], backtest_results["portfolio_returns"])
            plot_cumulative_returns(backtest_results["portfolio_returns"][selected_columns], backtest_paths["cumulative_returns"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_drawdowns(backtest_results["portfolio_returns"][selected_columns], backtest_paths["drawdown"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_scaled_log_returns(backtest_results["portfolio_returns"][selected_columns], backtest_paths["scaled_log_returns"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_all_methods_scaled_log_returns(backtest_results["portfolio_returns"], backtest_paths["all_methods_scaled_log_returns"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_family_vs_benchmarks(backtest_results["portfolio_returns"], "naive", backtest_paths["naive_vs_benchmarks"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_family_vs_benchmarks(backtest_results["portfolio_returns"], "mvo", backtest_paths["mvo_vs_benchmarks"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_family_vs_benchmarks(backtest_results["portfolio_returns"], "black_litterman", backtest_paths["black_litterman_vs_benchmarks"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            plot_family_vs_benchmarks(backtest_results["portfolio_returns"], "ridge", backtest_paths["ridge_vs_benchmarks"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            if selected_columns:
                primary_strategy = selected_columns[0] if selected_columns[0] not in {"SPY_buy_and_hold", "equal_weight_benchmark"} else (selected_columns[2] if len(selected_columns) > 2 else selected_columns[0])
                plot_rolling_sharpe(backtest_results["portfolio_returns"][primary_strategy], backtest_paths["rolling_sharpe_monthly"], annualized=False, show=show_plots, strategy_name=primary_strategy, transaction_cost_bps=transaction_cost_bps)
                plot_rolling_sharpe(backtest_results["portfolio_returns"][primary_strategy], backtest_paths["rolling_sharpe_annualized"], annualized=True, show=show_plots, strategy_name=primary_strategy, transaction_cost_bps=transaction_cost_bps)
            plot_control_vs_treatment_boxplots(backtest_results["metrics_table"], backtest_paths["boxplots"], show=show_plots, transaction_cost_bps=transaction_cost_bps)
            if backtest_results["forecast_mode"] == "both":
                plot_forecast_mode_comparison(backtest_results["metrics_table"], backtest_paths["forecast_mode_comparison"], show=show_plots, transaction_cost_bps=transaction_cost_bps)

    if plot:
        plot_pca_clusters(reduced_df, regimes_series, output_path=pca_plot_path, show=show_plots, default_plot_format=plot_format)
        plot_kmeans_regimes(reduced_df.copy(), full_regimes, output_path=timeline_plot_path, show=show_plots)

    return {
        "input_csv": input_path,
        "processed_path": processed_path,
        "reduced_path": reduced_path,
        "regimes_path": regimes_path,
        "pca_plot_path": pca_plot_path if plot else None,
        "timeline_plot_path": timeline_plot_path if plot else None,
        "backtest_paths": backtest_paths,
        "processed_data": processed_df,
        "reduced_data": reduced_df,
        "regimes": regimes_series,
        "regime_probabilities": probs_df,
        "transition_matrix": transition_matrix,
        "current_regime_probs": current_regime_probs,
        "next_regime_probs": next_regime_prob_series,
        "asset_prices": prices_df,
        "asset_returns": returns_df,
        "aligned_data": aligned_data,
        "backtest": backtest_results,
        "result_tables": result_tables,
        "pca_components": n_components,
        "pca_model": pca_model,
        "preprocessing": preprocessing_info,
        "kmeans": {
            "pred_isolation": pred_isolation,
            "pred_kmeans": pred_kmeans,
            "kmeans_centers": kmeans_centers,
            "index_least_freq": minority_cluster,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the transformation, clustering, and paper-style walk-forward backtest pipeline.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Raw input CSV. Defaults to the first raw CSV in the data directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for processed, reduced, regime, and backtest outputs.")
    parser.add_argument("--kernel", default=DEFAULT_KERNEL, help="Kernel for Kernel PCA, for example rbf, cosine, or poly.")
    parser.add_argument("--kernel-components", type=int, default=DEFAULT_KERNEL_COMPONENTS, help="Number of Kernel PCA components to keep.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Kernel coefficient for rbf/poly. Defaults to sklearn behaviour.")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE, help="Degree for polynomial kernel.")
    parser.add_argument("--coef0", type=float, default=DEFAULT_COEF0, help="Independent term for polynomial kernel.")
    parser.add_argument("--regime-count", type=int, default=DEFAULT_CLUSTER_COUNT, help="Number of cosine K-means regimes inside the majority cluster.")
    parser.add_argument("--trim-rows", type=int, default=DEFAULT_TRIM_ROWS, help="Rows to drop after transformations.")
    parser.add_argument("--outlier-method", choices=("global", "expanding", "locf"), default=DEFAULT_OUTLIER_METHOD, help="Outlier detection strategy. global uses full-dataset IQR (original, has look-ahead). expanding uses point-in-time expanding-window IQR, outliers become NaN. locf uses point-in-time IQR and replaces outliers with the last valid observation.")
    parser.add_argument("--imputation-method", choices=("em", "locf", "em_burnin"), default=DEFAULT_IMPUTATION_METHOD, help="Missing-value imputation strategy. em is the original EM algorithm (has look-ahead). locf is Last Observation Carried Forward (no leakage). em_burnin fits EM on the first --imputation-burn-in months then projects forward with frozen loadings.")
    parser.add_argument("--imputation-burn-in", type=int, default=DEFAULT_IMPUTATION_BURN_IN, help="Number of months used as the burn-in window when --imputation-method=em_burnin.")
    parser.add_argument("--plot-format", default=DEFAULT_PLOT_FORMAT, help="File format for saved regime plots.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively as well as saving them.")
    parser.add_argument("--no-plots", action="store_true", help="Disable saved plots.")
    parser.add_argument("--download-etfs", action="store_true", help="Download monthly ETF prices from Yahoo Finance and compute returns.")
    parser.add_argument("--backtest", action="store_true", help="Run the paper-style walk-forward backtest.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Walk-forward estimation window in months.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength.")
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS, help="One-way transaction cost in basis points applied to monthly traded notional.")
    parser.add_argument("--forecast-mode", choices=FORECAST_MODE_CHOICES, default=DEFAULT_FORECAST_MODE, help="How to use regime probabilities in forecasting: hard picks the top regime, soft blends by probabilities, both runs both variants for comparison.")
    parser.add_argument("--cash-asset", action="store_true", help="Enable the synthetic 0%%-return CASH asset. Disabled by default for the pure soft-regime specification.")
    parser.add_argument("--no-cash-asset", action="store_true", help="Keep the synthetic 0%%-return CASH asset disabled. This is the default and is kept for backward compatibility.")
    parser.add_argument("--fixed-overlay-exposure", type=float, default=DEFAULT_FIXED_OVERLAY_EXPOSURE, help="Fixed risky-asset exposure after portfolio construction. Default 1.0 means no fixed risk overlay.")
    parser.add_argument("--cash-ticker", default=DEFAULT_CASH_TICKER, help="Ticker/name used for the synthetic cash asset.")
    parser.add_argument("--dynamic-risk-overlay", action="store_true", default=DEFAULT_ENABLE_DYNAMIC_RISK_OVERLAY, help="Enable a drawdown-based dynamic risk overlay. This automatically enables synthetic 0%%-return cash for the residual allocation.")
    parser.add_argument("--overlay-lookback-months", type=int, default=DEFAULT_OVERLAY_LOOKBACK_MONTHS, help="Number of past monthly strategy returns used to compute recent drawdown for the dynamic overlay.")
    parser.add_argument("--overlay-soft-drawdown", type=float, default=DEFAULT_OVERLAY_SOFT_DRAWDOWN, help="Recent max drawdown threshold that reduces exposure to --overlay-soft-exposure.")
    parser.add_argument("--overlay-hard-drawdown", type=float, default=DEFAULT_OVERLAY_HARD_DRAWDOWN, help="Recent max drawdown threshold that reduces exposure to --overlay-hard-exposure.")
    parser.add_argument("--overlay-soft-exposure", type=float, default=DEFAULT_OVERLAY_SOFT_EXPOSURE, help="Risky-asset exposure after the soft drawdown threshold is breached.")
    parser.add_argument("--overlay-hard-exposure", type=float, default=DEFAULT_OVERLAY_HARD_EXPOSURE, help="Risky-asset exposure after the hard drawdown threshold is breached.")
    parser.add_argument("--overlay-good-probability-threshold", type=float, default=DEFAULT_OVERLAY_GOOD_PROBABILITY_THRESHOLD, help="If recent drawdown is bad but probability of the good-regime group is at least this value, the dynamic overlay immediately returns to 100%% exposure.")
    parser.add_argument("--overlay-good-regime-count", type=int, default=DEFAULT_OVERLAY_GOOD_REGIME_COUNT, help="Number of highest-ranked regimes treated as the good-regime group. Regimes are ranked every rebalance using expanding-window equal-weight ETF returns.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for Isolation Forest and Euclidean K-means.",
    )
    parser.add_argument("--etf-tickers", nargs="+", default=DEFAULT_ETF_TICKERS, help="ETF tickers to download from Yahoo Finance.")
    return parser