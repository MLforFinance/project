from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from ..models.Isolation import Isolation_Euclidean_KMeans
    from ..models.modified_Kmeans import plot_kmeans_regimes
except ImportError:  # pragma: no cover
    from models.Isolation import Isolation_Euclidean_KMeans
    from models.modified_Kmeans import plot_kmeans_regimes




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


def compute_transition_matrix(regime_state: pd.Series | pd.DataFrame, n_regimes: int) -> pd.DataFrame:
    """Estimate regime transition probabilities.

    If `regime_state` is a Series of hard labels, this is the usual hard
    transition count. If it is a DataFrame of soft regime probabilities, this
    computes expected transition counts using outer products p_t * p_{t+1}.
    """
    transitions = pd.DataFrame(0.0, index=range(n_regimes), columns=range(n_regimes))

    if isinstance(regime_state, pd.DataFrame):
        probs = regime_state.iloc[:, :n_regimes].to_numpy(dtype=float)
        for t in range(len(probs) - 1):
            p_t = renormalize_probabilities(probs[t])
            p_next = renormalize_probabilities(probs[t + 1])
            transitions += np.outer(p_t, p_next)
    else:
        regimes = regime_state
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
    regimes, pred_isolation, pred_kmeans, kmeans_centers, index_least_freq, probs = Isolation_Euclidean_KMeans(
        X_window,
        r=regime_count,
    )

    prob_columns = [f"regime_prob_{i}" for i in range(regime_count + 1)]

    return {
        "regimes": pd.Series(regimes, index=X_window.index, name="regime"),
        "probabilities": pd.DataFrame(probs, index=X_window.index, columns=prob_columns),
        "pred_isolation": pred_isolation,
        "pred_kmeans": pred_kmeans,
        "kmeans_centers": kmeans_centers,
        "index_least_freq": index_least_freq,
    }



__all__ = [
    "compute_transition_matrix",
    "renormalize_probabilities",
    "next_regime_probs",
    "compute_window_regime_state",
    "plot_pca_clusters",
    "plot_kmeans_regimes",
]
