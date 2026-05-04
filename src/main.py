from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from .data_processing.PCA import optimal_PCA
    from .data_processing.preprocessing import preprocess_fred_md
    from .models.modified_Kmeans import modified_KMeans
    from .models.utils import get_data, plot_regimes
    from .models.Isolation import Isolation
    from .models.UMAP import fit_umap
    from .models.regime_hmm import fit_gmm_hmm

except ImportError:  # pragma: no cover - supports direct script execution
    from data_processing.PCA import optimal_PCA
    from data_processing.preprocessing import preprocess_fred_md
    from models.modified_Kmeans import modified_KMeans
    from models.utils import get_data, plot_regimes
    from models.Isolation import Isolation
    from models.UMAP import fit_umap
    from models.regime_hmm import fit_gmm_hmm

DEFAULT_CLUSTER_COUNT = 5
DEFAULT_TARGET_VARIANCE = 0.95
DEFAULT_PLOT_FORMAT = "svg"
DEFAULT_TRIM_ROWS = None


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


def derive_output_paths(input_csv: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_csv.stem
    return (
        output_dir / f"{stem}_processed.csv",
        output_dir / f"{stem}_reduced.csv",
        output_dir / f"{stem}_regimes.csv",
    )


def derive_plot_paths(input_csv: Path, output_dir: Path, plot_format: str) -> tuple[Path, Path]:
    stem = input_csv.stem
    suffix = plot_format.lstrip('.')
    return (
        output_dir / f"{stem}_pca_regimes.{suffix}",
        output_dir / f"{stem}_timeline_regimes.{suffix}",
    )

def plot_clusters(reduced_df: pd.DataFrame, regimes: pd.Series, output_path: Path | None = None, show: bool = False) -> None:
    if reduced_df.shape[1] < 3:
        return

    plot_df = reduced_df.iloc[:, :3].copy()
    plot_df["regime"] = regimes.to_numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        plot_df.iloc[:, 0],
        plot_df.iloc[:, 1],
        plot_df.iloc[:, 2],
        c=plot_df["regime"],
        cmap="tab10",
        alpha=0.8,
    )
    
    ax.set_xlabel(plot_df.columns[0])
    ax.set_ylabel(plot_df.columns[1])
    ax.set_zlabel(plot_df.columns[2])
    ax.set_title("Regimes in 3D Reduced Space")
    ax.legend(*scatter.legend_elements(), title="Regime")
    
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=output_path.suffix.lstrip('.') or DEFAULT_PLOT_FORMAT, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def run_pipeline(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_variance: float = DEFAULT_TARGET_VARIANCE,
    regime_count: int = DEFAULT_CLUSTER_COUNT,
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

    processed_path, reduced_path, regimes_path = derive_output_paths(
        input_path, output_dir_path)
    pca_plot_path, timeline_plot_path = derive_plot_paths(
        input_path, output_dir_path, plot_format)

    processed_df, preprocessing_info = preprocess_fred_md(
        input_path,
        processed_path,
        trim_rows=trim_rows,
    )

    reduced_df, n_components, pca_model = optimal_PCA(
        processed_df,
        target_variance=target_variance,
        plot=True,
    )
    reduced_df.to_csv(reduced_path)

    regimes, pred_l2, pred_cos, l2_centers, cosine_centers, minority_cluster = modified_KMeans(
        reduced_df,
        r=regime_count,
    )

    regimes_df = reduced_df.copy()
    regimes_df["regime"] = regimes
    regimes_df.to_csv(regimes_path)

    regimes_series = pd.Series(regimes, index=reduced_df.index, name="regime")

    if plot:
        plot_clusters(reduced_df, regimes_series,
                          output_path=pca_plot_path, show=show_plots)
        plot_regimes(reduced_df.copy(), regimes,
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
            "pred_l2": pred_l2,
            "pred_cos": pred_cos,
            "l2_centers": l2_centers,
            "cosine_centers": cosine_centers,
            "minority_cluster": minority_cluster,
        },
    }


def run_hmm_pipeline(
    input_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    regime_count: int = DEFAULT_CLUSTER_COUNT,
    trim_rows: int | None = DEFAULT_TRIM_ROWS,
    plot: bool = True,
    plot_format: str = DEFAULT_PLOT_FORMAT,
    show_plots: bool = False,
) -> dict:
    
    data_dir = default_data_dir()
    input_path = Path(input_csv) if input_csv is not None else discover_input_csv(data_dir)
    output_dir_path = Path(output_dir) if output_dir is not None else input_path.parent
    output_dir_path.mkdir(parents=True, exist_ok=True)

    processed_path, reduced_path, regimes_path = derive_output_paths(input_path, output_dir_path)
    umap_plot_path, timeline_plot_path = derive_plot_paths(input_path, output_dir_path, plot_format)

    processed_df, preprocessing_info = preprocess_fred_md(
        input_path,
        processed_path,
        trim_rows=trim_rows,
    )
    
    # Anomaly Detection (Regime 0)
    regime0, normal, r = Isolation(processed_df, n_estimators=200, contamination=0.1)

    reduced_df, umap_model = fit_umap(
        normal,
        n_components=6,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        epochs = 500
    )
    reduced_df.to_csv(reduced_path)
    
    # GMM-HMM Clustering
    states_series, state_probs_df, hmm_model = fit_gmm_hmm(reduced_df, n_components=regime_count)
    
    states_series = states_series + 1
    
    full_regimes = pd.Series(index=processed_df.index, dtype=int, name="regime")
    full_regimes.loc[regime0.index] = 0
    full_regimes.loc[normal.index] = states_series
    
    prob_cols = [f"prob_regime_{i}" for i in range(regime_count + 1)]
    full_probs = pd.DataFrame(index=processed_df.index, columns=prob_cols, dtype=float)
    
    full_probs.loc[regime0.index, "prob_regime_0"] = 1.0
    full_probs.loc[regime0.index, full_probs.columns[1:]] = 0.0
    
    full_probs.loc[normal.index, "prob_regime_0"] = 0.0
    for i in range(regime_count):
        full_probs.loc[normal.index, f"prob_regime_{i+1}"] = state_probs_df.iloc[:, i].values

    regimes_df = processed_df.copy()
    regimes_df["regime"] = full_regimes
    regimes_df = pd.concat([regimes_df, full_probs], axis=1)
    regimes_df.to_csv(regimes_path)

    if plot:
        plot_clusters(reduced_df, states_series, output_path=umap_plot_path, show=show_plots)
        plot_regimes(processed_df.copy(), full_regimes.values, output_path=timeline_plot_path, show=show_plots)
        
    return {
        "input_csv": input_path,
        "processed_path": processed_path,
        "reduced_path": reduced_path,
        "regimes_path": regimes_path,
        "umap_plot_path": umap_plot_path if plot else None,
        "timeline_plot_path": timeline_plot_path if plot else None,
        "processed_data": processed_df,
        "reduced_data": reduced_df,
        "regimes": full_regimes,
        "probabilities": full_probs,
        "umap_model": umap_model,
        "hmm_model": hmm_model,
        "preprocessing": preprocessing_info,
        "anomaly_ratio": r
    }
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full transformation, PCA, and clustering pipeline.")
    parser.add_argument("--pipeline", type=str, default = "baseline", choices=["baseline", "hmm"],
                        help="Choose ur pipeline to run")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="Raw input CSV. Defaults to the first raw CSV in the data directory.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for processed, reduced, and regime CSV outputs.")
    parser.add_argument("--target-variance", type=float, default=DEFAULT_TARGET_VARIANCE,
                        help="Target cumulative explained variance for PCA.")
    parser.add_argument("--regime-count", type=int, default=DEFAULT_CLUSTER_COUNT,
                        help="Number of cosine K-means regimes inside the majority cluster.")
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
    if args.pipeline == 'baseline':
        results = run_pipeline(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            target_variance=args.target_variance,
            regime_count=args.regime_count,
            trim_rows=args.trim_rows,
            plot=not args.no_plots,
            plot_format=args.plot_format,
            show_plots=args.show_plots
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
        print(f"PCA components selected: {results['pca_components']}")

    elif args.pipeline == "hmm":
        results = run_hmm_pipeline(
            input_csv = args.input_csv,
            output_dir = args.output_dir,
            regime_count = args.regime_count,
            trim_rows=args.trim_rows,
            plot= not args.no_plots,
            plot_format=args.plot_format,
            show_plots=args.show_plots
        )


if __name__ == "__main__":
    main()
