from __future__ import annotations

import argparse
from email import parser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .data_processing.kernel_PCA import optimal_kernel_PCA
    from .data_processing.preprocessing import preprocess_fred_md
    from .models.Isolation import Isolation_Euclidean_KMeans
    from .models.modified_Kmeans import plot_kmeans_regimes
except ImportError:  # pragma: no cover - supports direct script execution
    from data_processing.kernel_PCA import optimal_kernel_PCA
    from data_processing.preprocessing import preprocess_fred_md
    from models.Isolation import Isolation_Euclidean_KMeans
    from models.modified_Kmeans import plot_kmeans_regimes


DEFAULT_KERNEL = "poly"
DEFAULT_KERNEL_COMPONENTS = 6
DEFAULT_GAMMA = None
DEFAULT_DEGREE = 2
DEFAULT_COEF0 = 1.0

DEFAULT_RANDOM_STATE = 42

DEFAULT_CLUSTER_COUNT = 5
DEFAULT_PLOT_FORMAT = "svg"
DEFAULT_TRIM_ROWS = None

NBER_RECESSIONS_FROM_1960 = [
    ("1960-05-01", "1961-02-01"),
    ("1970-01-01", "1970-11-01"),
    ("1973-12-01", "1975-03-01"),
    ("1980-02-01", "1980-07-01"),
    ("1981-08-01", "1982-11-01"),
    ("1990-08-01", "1991-03-01"),
    ("2001-04-01", "2001-11-01"),
    ("2008-01-01", "2009-06-01"),
    ("2020-03-01", "2020-04-01"),
]

DEFAULT_TRIM_ROWS = None
KEY_FEATURES = [
    "UNRATE",
    "UMCSENTx",
    "FEDFUNDS",
    "RPI",
    "CPIAUCSL",
    "S&P 500",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    return project_root() / "data"


def discover_input_csv(data_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in data_dir.glob("*.csv")
        if not path.stem.endswith(("_processed", "_reduced", "_regimes"))
    )
    if not candidates:
        raise FileNotFoundError(f"No raw CSV files found in {data_dir}")
    return candidates[0]

def derive_output_paths(input_csv: Path, output_dir: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    stem = input_csv.stem
    return (
        output_dir / f"{stem}_processed.csv",
        output_dir / f"{stem}_reduced.csv",
        output_dir / f"{stem}_regimes.csv",
        output_dir / f"{stem}_macro_regime_means.csv",
        output_dir / f"{stem}_macro_regime_means.html",
        output_dir / f"{stem}_nber_recession_means.html",
    )




def derive_plot_paths(input_csv: Path, output_dir: Path, plot_format: str) -> tuple[Path, Path]:
    stem = input_csv.stem
    suffix = plot_format.lstrip('.')
    return (
        output_dir / f"{stem}_pca_regimes.{suffix}",
        output_dir / f"{stem}_timeline_regimes.{suffix}",
    )


def plot_pca_clusters(reduced_df: pd.DataFrame, regimes: pd.Series, output_path: Path | None = None, show: bool = False) -> None:
    if reduced_df.shape[1] < 2:
        return

    plot_df = reduced_df.iloc[:, :2].copy()
    plot_df["regime"] = regimes.to_numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        plot_df.iloc[:, 0],
        plot_df.iloc[:, 1],
        c=plot_df["regime"],
        cmap="tab10",
        alpha=0.8,
    )
    ax.set_xlabel(plot_df.columns[0])
    ax.set_ylabel(plot_df.columns[1])
    ax.set_title("K-means regimes in PCA space")
    ax.legend(*scatter.legend_elements(), title="Regime")
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=output_path.suffix.lstrip(
            '.') or DEFAULT_PLOT_FORMAT, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_macro_regime_means(
    original_df_aligned: pd.DataFrame,
    regimes: pd.Series | list | tuple,
    key_features: list[str],
) -> pd.DataFrame:
    missing = [col for col in key_features if col not in original_df_aligned.columns]

    if missing:
        print(f"Warning: missing columns skipped: {missing}")

    available_features = [
        col for col in key_features
        if col in original_df_aligned.columns
    ]

    if not available_features:
        raise ValueError("None of the requested key features exist in processed data.")

    temp = original_df_aligned[available_features].copy()
    temp["regime"] = list(regimes)

    macro_summary = temp.groupby("regime", as_index=False).mean()
    macro_summary = macro_summary.sort_values("regime").reset_index(drop=True)

    macro_summary = macro_summary.round(3)

    return macro_summary

def build_nber_recession_means(
    original_df: pd.DataFrame,
    recession_periods: list[tuple[str, str]],
    key_features: list[str],
) -> pd.DataFrame:
    missing = [col for col in key_features if col not in original_df.columns]

    if missing:
        print(f"Warning: missing columns skipped for NBER recession table: {missing}")

    available_features = [
        col for col in key_features
        if col in original_df.columns
    ]

    if not available_features:
        raise ValueError("None of the requested key features exist in processed data.")

    df = original_df.copy()
    df.index = pd.to_datetime(df.index)

    recession_mask = pd.Series(False, index=df.index)

    for start, end in recession_periods:
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        recession_mask |= (df.index >= start_date) & (df.index <= end_date)

    recession_df = df.loc[recession_mask, available_features]

    if recession_df.empty:
        raise ValueError("No observations found inside the NBER recession periods.")

    recession_summary = pd.DataFrame([recession_df.mean()])
    recession_summary = recession_summary.round(3)

    return recession_summary

def get_full_processed_for_tables(
    input_path: Path,
    trim_rows: int | None,
) -> pd.DataFrame:
    full_processed_df, _ = preprocess_fred_md(
        input_path,
        output_path=None,
        trim_rows=trim_rows,
        excluded_columns=(),
    )
    return full_processed_df

def run_pipeline(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    kernel: str = DEFAULT_KERNEL,
    kernel_components: int = DEFAULT_KERNEL_COMPONENTS,
    gamma: float | None = DEFAULT_GAMMA,
    degree: int = DEFAULT_DEGREE,
    coef0: float = DEFAULT_COEF0,
    regime_count: int = DEFAULT_CLUSTER_COUNT,
    random_state: int = DEFAULT_RANDOM_STATE,
    trim_rows: int | None = DEFAULT_TRIM_ROWS,
    plot: bool = True,
    plot_format: str = DEFAULT_PLOT_FORMAT,
    show_plots: bool = False,
) -> dict:
    data_dir = default_data_dir()
    input_path = Path(
        input_csv) if input_csv is not None else discover_input_csv(data_dir)
    output_dir_path = Path(
        output_dir) if output_dir is not None else input_path.parent
    output_dir_path.mkdir(parents=True, exist_ok=True)

    processed_path, reduced_path, regimes_path, macro_summary_path, macro_summary_html_path, nber_recession_html_path = derive_output_paths(
    input_path, output_dir_path)

    pca_plot_path, timeline_plot_path = derive_plot_paths(
        input_path, output_dir_path, plot_format)

    processed_df, preprocessing_info = preprocess_fred_md(
        input_path,
        processed_path,
        trim_rows=trim_rows,
    )

    table_processed_df = get_full_processed_for_tables(
        input_path=input_path,
        trim_rows=trim_rows,
    )

    reduced_df, n_components, pca_model = optimal_kernel_PCA(   
        processed_df,
        kernel=kernel,
        n_components=kernel_components,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
    )

  
    reduced_df.to_csv(reduced_path)

    regimes, pred_isolation, pred_kmeans, kmeans_centers, index_least_freq = Isolation_Euclidean_KMeans(
        reduced_df,
        r=regime_count,
        random_state=random_state,
    )
    
    regimes_df = reduced_df.copy()
    regimes_df["regime"] = regimes
    regimes_df.to_csv(regimes_path)


    regimes_series = pd.Series(regimes, index=reduced_df.index, name="regime")

    macro_summary = build_macro_regime_means(
        original_df_aligned=table_processed_df.loc[reduced_df.index],
        regimes=regimes_series,
        key_features=KEY_FEATURES,
    )

    macro_summary.to_csv(macro_summary_path, index=False)

    # Rename columns for cleaner display in the table
    macro_summary_display = macro_summary.rename(columns={
        "regime": "Regime",
        "UNRATE": "Unemployment Rate",
        "UMCSENTx": "Consumer Sentiment",
        "FEDFUNDS": "Federal Funds Rate",
        "RPI": "Real Personal Income",
        "CPIAUCSL": "Consumer Price Index",
        "S&P 500": "S&P 500",
    })


    def highlight_high_low(column):
        styles = [""] * len(column)

        if column.name == "Regime":
            return styles

        max_value = column.max()
        min_value = column.min()

        for i, value in enumerate(column):
            if value == max_value:
                styles[i] = "background-color: #f4cccc;"  # light red
            elif value == min_value:
                styles[i] = "background-color: #cfe2f3;"  # light blue

        return styles

    styled_macro_summary = (
        macro_summary_display.style
        .hide(axis="index")
        .apply(highlight_high_low, axis=0)
        .format(precision=3)
        .set_properties(**{
            "text-align": "center",
            "vertical-align": "middle",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
        ])
    )

    styled_macro_summary.to_html(macro_summary_html_path)

    nber_recession_summary = build_nber_recession_means(
        original_df=table_processed_df,
        recession_periods=NBER_RECESSIONS_FROM_1960,
        key_features=KEY_FEATURES,
    )
    
    
    nber_recession_summary_display = nber_recession_summary.rename(columns={
        "UNRATE": "Unemployment Rate",
        "UMCSENTx": "Consumer Sentiment",
        "FEDFUNDS": "Federal Funds Rate",
        "RPI": "Real Personal Income",
        "CPIAUCSL": "Consumer Price Index",
        "S&P 500": "S&P 500",
    })

    styled_nber_recession_summary = (
        nber_recession_summary_display.style
        .hide(axis="index")
        .format(precision=3)
        .set_properties(**{
            "text-align": "center",
            "vertical-align": "middle",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("text-align", "center"),
                    ("vertical-align", "middle"),
                ],
            },
        ])
    )

    styled_nber_recession_summary.to_html(nber_recession_html_path)

    if plot:
        plot_pca_clusters(reduced_df, regimes_series,
                          output_path=pca_plot_path, show=show_plots)
        plot_kmeans_regimes(reduced_df.copy(), regimes,recessions=NBER_RECESSIONS_FROM_1960,
                            output_path=timeline_plot_path, show=show_plots)

    return {
        "input_csv": input_path,
        "processed_path": processed_path,
        "reduced_path": reduced_path,
        "regimes_path": regimes_path,
        "pca_plot_path": pca_plot_path if plot else None,
        "timeline_plot_path": timeline_plot_path if plot else None,
        "processed_data": processed_df,
        "reduced_data": reduced_df,
        "regimes": regimes_series,
        "pca_components": n_components,
        "pca_model": pca_model,
        "preprocessing": preprocessing_info,
        "kmeans": {
            "pred_isolation": pred_isolation,
            "pred_kmeans": pred_kmeans,
            "kmeans_centers": kmeans_centers,
            "index_least_freq": index_least_freq,
        },
        "macro_summary_path": macro_summary_path,
        "macro_summary_html_path": macro_summary_html_path,
        "macro_summary": macro_summary,
        "nber_recession_html_path": nber_recession_html_path,
        "nber_recession_summary": nber_recession_summary,
    }



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full transformation, PCA, and clustering pipeline.")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="Raw input CSV. Defaults to the first raw CSV in the data directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for processed, reduced, and regime CSV outputs.")
    parser.add_argument("--kernel", default=DEFAULT_KERNEL,
                    help="Kernel for Kernel PCA, for example rbf, cosine, or poly.")
    parser.add_argument("--kernel-components", type=int, default=DEFAULT_KERNEL_COMPONENTS,
                    help="Number of Kernel PCA components to keep.")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                    help="Kernel coefficient for rbf/poly. Defaults to sklearn behaviour.")
    parser.add_argument("--degree", type=int, default=DEFAULT_DEGREE,
                    help="Degree for polynomial kernel.")
    parser.add_argument("--coef0", type=float, default=DEFAULT_COEF0,
                    help="Independent term for polynomial kernel.")
    parser.add_argument("--regime-count", type=int, default=DEFAULT_CLUSTER_COUNT,
                        help="Number of cosine K-means regimes inside the majority cluster.")

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed for Isolation Forest and Euclidean K-means.",
    )

    parser.add_argument("--trim-rows", type=int, default=DEFAULT_TRIM_ROWS,
                        help="Rows to drop after transformations. Defaults to automatic inference from the FRED transformation codes.")
    parser.add_argument("--plot-format", default=DEFAULT_PLOT_FORMAT,
                        help="File format for saved plots, for example svg or png.")
    parser.add_argument("--show-plots", action="store_true",
                        help="Attempt to display plots interactively as well as saving them.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Disable PCA and clustering visualizations.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        kernel=args.kernel,
        kernel_components=args.kernel_components,
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
        regime_count=args.regime_count,
        random_state=args.random_state,
        trim_rows=args.trim_rows,
        plot=not args.no_plots,
        plot_format=args.plot_format,
        show_plots=args.show_plots,
    )
    

    print(f"Input CSV: {results['input_csv']}")
    print(f"Processed data saved to: {results['processed_path']}")
    print(f"Reduced PCA data saved to: {results['reduced_path']}")
    print(f"Regime assignments saved to: {results['regimes_path']}")
    if results["pca_plot_path"] is not None:
        print(f"PCA regime plot saved to: {results['pca_plot_path']}")
    if results["timeline_plot_path"] is not None:
        print(
            f"Timeline regime plot saved to: {results['timeline_plot_path']}")
    # print(
    #     f"Transformation burn-in rows removed: {results['preprocessing']['burn_in_rows']}")
    # print(
    #     f"Missing values after prepare_missing: {results['preprocessing']['n_missing_after_prepare_missing']}")
    # print(
    #     f"Missing values after remove_outliers: {results['preprocessing']['n_missing_after_remove_outliers']}")
    # print(
    #     f"Outliers converted to NaN: {results['preprocessing']['n_outliers_total']}")
    # print(
    #     f"Preprocessed matrix shape: {results['preprocessing']['output_shape']}")
    print(f"Kernel PCA components selected: {results['pca_components']}")
    print(f"Macro regime means saved to: {results['macro_summary_path']}")
    print(f"Colored macro regime means saved to: {results['macro_summary_html_path']}")
    print(f"NBER recession means saved to: {results['nber_recession_html_path']}")
if __name__ == "__main__":
    main()
