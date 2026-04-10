from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data"


def discover_input_csv(data_dir: Path) -> Path:
    excluded_suffixes = (
        "_processed",
        "_reduced",
        "_regimes",
        "_etf_prices",
        "_etf_returns",
        "_portfolio_returns",
        "_weights",
        "_predictions",
        "_strategy_metrics",
        "_naive_result_table",
        "_bl_mvo_result_table",
        "_ridge_result_table",
    )
    candidates = sorted(path for path in data_dir.glob("*.csv") if not path.stem.endswith(excluded_suffixes))
    if not candidates:
        raise FileNotFoundError(f"No raw CSV files found in {data_dir}")
    return candidates[0]


def derive_output_paths(input_csv: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_csv.stem
    return (
        output_dir / f"{stem}_processed.csv",
        output_dir / f"{stem}_reduced.csv",
        output_dir / f"{stem}_regimes.csv",
    )


def derive_plot_paths(input_csv: Path, output_dir: Path, plot_format: str) -> tuple[Path, Path]:
    stem = input_csv.stem
    suffix = plot_format.lstrip(".")
    return (
        output_dir / f"{stem}_pca_regimes.{suffix}",
        output_dir / f"{stem}_timeline_regimes.{suffix}",
    )


def derive_backtest_paths(input_csv: Path, output_dir: Path) -> dict[str, Path]:
    stem = input_csv.stem
    return {
        "etf_prices": output_dir / f"{stem}_etf_prices.csv",
        "etf_returns": output_dir / f"{stem}_etf_returns.csv",
        "portfolio_returns": output_dir / f"{stem}_portfolio_returns.csv",
        "weights": output_dir / f"{stem}_weights.csv",
        "predictions": output_dir / f"{stem}_predictions.csv",
        "metrics": output_dir / f"{stem}_metrics.json",
        "metrics_table": output_dir / f"{stem}_strategy_metrics.csv",
        "naive_result_table": output_dir / f"{stem}_naive_result_table.csv",
        "bl_mvo_result_table": output_dir / f"{stem}_bl_mvo_result_table.csv",
        "ridge_result_table": output_dir / f"{stem}_ridge_result_table.csv",
        "cumulative_returns": output_dir / f"{stem}_cumulative_returns.png",
        "drawdown": output_dir / f"{stem}_drawdown.png",
        "scaled_log_returns": output_dir / f"{stem}_scaled_log_returns.png",
        "rolling_sharpe_monthly": output_dir / f"{stem}_rolling_sharpe_monthly.png",
        "rolling_sharpe_annualized": output_dir / f"{stem}_rolling_sharpe_annualized.png",
        "all_methods_scaled_log_returns": output_dir / f"{stem}_all_methods_scaled_log_returns.png",
        "naive_vs_benchmarks": output_dir / f"{stem}_naive_vs_benchmarks.png",
        "mvo_vs_benchmarks": output_dir / f"{stem}_mvo_vs_benchmarks.png",
        "black_litterman_vs_benchmarks": output_dir / f"{stem}_black_litterman_vs_benchmarks.png",
        "ridge_vs_benchmarks": output_dir / f"{stem}_ridge_vs_benchmarks.png",
        "boxplots": output_dir / f"{stem}_control_vs_treatment_boxplots.png",
    }
