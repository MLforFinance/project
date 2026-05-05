from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from ..models.modified_Kmeans import modified_KMeans, plot_kmeans_regimes
    from ..models.Isolation import Isolation_UMAP_HMM
except ImportError:  # pragma: no cover
    from models.modified_Kmeans import modified_KMeans, plot_kmeans_regimes
    from models.Isolation import Isolation_UMAP_HMM


def plot_pca_clusters(reduced_df: pd.DataFrame, regimes: pd.Series, output_path: Path | None = None, show: bool = False, default_plot_format: str = "svg") -> None:
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
        fig.savefig(output_path, format=output_path.suffix.lstrip(".") or default_plot_format, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def compute_transition_matrix(regimes: pd.Series, n_regimes: int) -> pd.DataFrame:
    transitions = pd.DataFrame(0.0, index=range(n_regimes), columns=range(n_regimes))
    for t in range(len(regimes) - 1):
        i = int(regimes.iloc[t])
        j = int(regimes.iloc[t + 1])
        transitions.loc[i, j] += 1.0
    return transitions.div(transitions.sum(axis=1), axis=0).fillna(0.0)


def renormalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    norm = np.sum(np.abs(probs))
    if norm <= 0:
        return np.repeat(1.0 / len(probs), len(probs))
    return probs / norm


def next_regime_probs(current_probs: np.ndarray, transition_matrix: np.ndarray) -> np.ndarray:
    return renormalize_probabilities(current_probs) @ np.asarray(transition_matrix, dtype=float)


def compute_window_regime_state(X_window: pd.DataFrame, regime_count: int) -> dict[str, pd.DataFrame | pd.Series]:
    regimes, probs, pred_l2, pred_cos, l2_centers, cosine_centers, minority_cluster = modified_KMeans(X_window, r=regime_count)
    prob_columns = [f"regime_prob_{i}" for i in range(regime_count + 1)]
    return {
        "regimes": pd.Series(regimes, index=X_window.index, name="regime"),
        "probabilities": pd.DataFrame(probs, index=X_window.index, columns=prob_columns),
        "pred_l2": pred_l2,
        "pred_cos": pred_cos,
        "l2_centers": l2_centers,
        "cosine_centers": cosine_centers,
        "minority_cluster": minority_cluster,
    }


def compute_window_regime_state_hmm(
    X_window: pd.DataFrame,
    regime_count: int,
    prob_mode: str = "soft",
    umap_components: int = 4,
    umap_n_neighbors: int = 15,
    umap_epochs: int = 500,
    iso_score_scale: float = 5.0,
) -> dict[str, pd.DataFrame | pd.Series]:
    regimes, probs, pred_isolation, umap_reduced, hmm_states_full, umap_mapper, hmm_model, anomaly_mask = Isolation_UMAP_HMM(
        X_window,
        r=regime_count,
        prob_mode=prob_mode,
        umap_components=umap_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_epochs=umap_epochs,
        iso_score_scale=iso_score_scale,
    )
    prob_columns = [f"regime_prob_{i}" for i in range(regime_count + 1)]
    return {
        "regimes": pd.Series(regimes, index=X_window.index, name="regime"),
        "probabilities": pd.DataFrame(probs, index=X_window.index, columns=prob_columns),
        "pred_isolation": pred_isolation,
        "umap_reduced": umap_reduced,
        "hmm_states_full": hmm_states_full,
        "umap_mapper": umap_mapper,
        "hmm_model": hmm_model,
        "anomaly_mask": anomaly_mask,
    }


__all__ = [
    "compute_transition_matrix",
    "renormalize_probabilities",
    "next_regime_probs",
    "compute_window_regime_state",
    "compute_window_regime_state_hmm",
    "plot_pca_clusters",
    "plot_kmeans_regimes",
]
