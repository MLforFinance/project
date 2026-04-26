from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from ..data_processing.PCA import optimal_PCA
    from ..data_processing.preprocessing import preprocess_fred_md
except ImportError:  # pragma: no cover
    from data_processing.PCA import optimal_PCA
    from data_processing.preprocessing import preprocess_fred_md

from .backtest import run_walk_forward_backtest
from .config import (
    DEFAULT_CLUSTER_COUNT,
    DEFAULT_ETF_TICKERS,
    DEFAULT_PLOT_FORMAT,
    DEFAULT_TARGET_VARIANCE,
    DEFAULT_TRANSACTION_COST_BPS,
    DEFAULT_TRIM_ROWS,
    DEFAULT_WINDOW_SIZE,
)
from .data_sources import align_macro_and_returns, download_etf_prices, ensure_month_start_index, prices_to_returns
from .paths import default_data_dir, derive_backtest_paths, derive_output_paths, derive_plot_paths, discover_input_csv
from .regime_pipeline import compute_transition_matrix, next_regime_probs, plot_kmeans_regimes, plot_pca_clusters, renormalize_probabilities
try:
    from ..models.modified_Kmeans import modified_KMeans
except ImportError:  # pragma: no cover
    from models.modified_Kmeans import modified_KMeans
from .reporting import build_result_tables, plot_all_methods_scaled_log_returns, plot_control_vs_treatment_boxplots, plot_cumulative_returns, plot_drawdowns, plot_family_vs_benchmarks, plot_rolling_sharpe, plot_scaled_log_returns, select_plot_columns


def run_pipeline(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_variance: float = DEFAULT_TARGET_VARIANCE,
    regime_count: int = DEFAULT_CLUSTER_COUNT,
    trim_rows: int | None = DEFAULT_TRIM_ROWS,
    plot: bool = True,
    plot_format: str = DEFAULT_PLOT_FORMAT,
    show_plots: bool = False,
    ridge_alpha: float = 1.0,
    etf_tickers: list[str] | tuple[str, ...] = DEFAULT_ETF_TICKERS,
    download_etfs: bool = False,
    backtest: bool = False,
    window_size: int = DEFAULT_WINDOW_SIZE,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    asset_returns_df: pd.DataFrame | None = None,
) -> dict[str, object]:
    data_dir = default_data_dir()
    input_path = Path(input_csv) if input_csv is not None else discover_input_csv(data_dir)
    output_dir_path = Path(output_dir) if output_dir is not None else input_path.parent
    output_dir_path.mkdir(parents=True, exist_ok=True)

    processed_path, reduced_path, regimes_path = derive_output_paths(input_path, output_dir_path)
    pca_plot_path, timeline_plot_path = derive_plot_paths(input_path, output_dir_path, plot_format)
    backtest_paths = derive_backtest_paths(input_path, output_dir_path)

    processed_df, preprocessing_info = preprocess_fred_md(input_path, processed_path, trim_rows=trim_rows)
    processed_df = ensure_month_start_index(processed_df)

    reduced_df, n_components, pca_model = optimal_PCA(processed_df, target_variance=target_variance, plot=False)
    reduced_df = ensure_month_start_index(reduced_df)
    reduced_df.to_csv(reduced_path)

    full_regimes, full_probs, pred_l2, pred_cos, l2_centers, cosine_centers, minority_cluster = modified_KMeans(reduced_df, r=regime_count)
    prob_columns = [f"regime_prob_{i}" for i in range(regime_count + 1)]
    regimes_series = pd.Series(full_regimes, index=reduced_df.index, name="regime")
    probs_df = pd.DataFrame(full_probs, index=reduced_df.index, columns=prob_columns)
    pd.concat([reduced_df, regimes_series, probs_df], axis=1).to_csv(regimes_path)

    transition_matrix = compute_transition_matrix(regimes_series, regime_count + 1)
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
        aligned_data = align_macro_and_returns(reduced_df, returns_df)

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
        )
        backtest_results["portfolio_returns"].to_csv(backtest_paths["portfolio_returns"])
        backtest_results["gross_portfolio_returns"].to_csv(backtest_paths["gross_portfolio_returns"])
        backtest_results["turnover"].to_csv(backtest_paths["turnover"])
        backtest_results["transaction_costs"].to_csv(backtest_paths["transaction_costs"])
        backtest_results["weights"].to_csv(backtest_paths["weights"])
        backtest_results["predictions"].to_csv(backtest_paths["predictions"])
        backtest_results["metrics_table"].to_csv(backtest_paths["metrics_table"], index=False)
        metrics_payload = {
            "transaction_cost_bps": backtest_results["transaction_cost_bps"],
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
            "pred_l2": pred_l2,
            "pred_cos": pred_cos,
            "l2_centers": l2_centers,
            "cosine_centers": cosine_centers,
            "minority_cluster": minority_cluster,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the transformation, clustering, and paper-style walk-forward backtest pipeline.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Raw input CSV. Defaults to the first raw CSV in the data directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for processed, reduced, regime, and backtest outputs.")
    parser.add_argument("--target-variance", type=float, default=DEFAULT_TARGET_VARIANCE, help="Target cumulative explained variance for PCA.")
    parser.add_argument("--regime-count", type=int, default=DEFAULT_CLUSTER_COUNT, help="Number of cosine K-means regimes inside the majority cluster.")
    parser.add_argument("--trim-rows", type=int, default=DEFAULT_TRIM_ROWS, help="Rows to drop after transformations.")
    parser.add_argument("--plot-format", default=DEFAULT_PLOT_FORMAT, help="File format for saved regime plots.")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively as well as saving them.")
    parser.add_argument("--no-plots", action="store_true", help="Disable saved plots.")
    parser.add_argument("--download-etfs", action="store_true", help="Download monthly ETF prices from Yahoo Finance and compute returns.")
    parser.add_argument("--backtest", action="store_true", help="Run the paper-style walk-forward backtest.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE, help="Walk-forward estimation window in months.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Ridge regularization strength.")
    parser.add_argument("--transaction-cost-bps", type=float, default=DEFAULT_TRANSACTION_COST_BPS, help="One-way transaction cost in basis points applied to monthly traded notional.")
    parser.add_argument("--etf-tickers", nargs="+", default=DEFAULT_ETF_TICKERS, help="ETF tickers to download from Yahoo Finance.")
    return parser
