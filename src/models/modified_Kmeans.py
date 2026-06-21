import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

RANDOM_SEED = 42
KMEANS_ITER = 10


def compute_cluster_probabilities(distances: np.ndarray) -> np.ndarray:
    distance_sums = distances.sum(axis=1, keepdims=True)
    base_scores = 1.0 - distances / (distance_sums + 1e-12)
    base_scores = np.clip(base_scores, 0.0, None)
    score_sums = base_scores.sum(axis=1, keepdims=True)

    uniform = np.full_like(base_scores, 1.0 / base_scores.shape[1])
    return np.divide(base_scores, score_sums, out=uniform, where=score_sums > 0)


def combine_regime_probabilities(regime_zero_probs: np.ndarray, cosine_probs: np.ndarray) -> np.ndarray:
    pmax = np.max(cosine_probs, axis=1)
    regime_zero_scores = -pmax * np.log2(np.clip(1.0 - regime_zero_probs, 1e-12, None))
    raw = np.column_stack([regime_zero_scores, cosine_probs])
    raw_sums = raw.sum(axis=1, keepdims=True)
    uniform = np.full_like(raw, 1.0 / raw.shape[1])
    return np.divide(raw, raw_sums, out=uniform, where=raw_sums > 0)


def modified_KMeans(data: pd.DataFrame, r: int = 5):
    data_array = np.asarray(data)
    l2_norms = np.linalg.norm(data_array, axis=1, keepdims=True)

    model_l2 = KMeans(
        n_clusters=2,
        tol=1e-5,
        random_state=RANDOM_SEED,
        n_init=KMEANS_ITER,
    )
    pred_l2 = model_l2.fit_transform(l2_norms)
    l2_probs = compute_cluster_probabilities(pred_l2)

    mask_1 = np.argmin(pred_l2, axis=1) == 1
    count_1 = np.sum(mask_1)
    count_0 = len(data_array) - count_1

    if count_0 > count_1:
        index_least_freq = 1
        majority_mask = ~mask_1
    else:
        index_least_freq = 0
        majority_mask = mask_1

    final_regimes = np.zeros(len(data_array), dtype=int)
    data_to_split = data_array[majority_mask]
    cosine_probs_full = np.full((len(data_array), r), 1.0 / max(r, 1), dtype=float)
    pred_cos = np.empty((len(data_to_split), 0))
    centroids_cos = np.empty((0, data_array.shape[1]))

    if len(data_to_split) > 0:
        k_eff = min(r, len(data_to_split))
        labels_cos, centroids_cos_eff, pred_cos = KMeansCosine_multi(
            data_to_split,
            k=k_eff,
            epsilon=1e-4,
            n_init=KMEANS_ITER,
            random_state=RANDOM_SEED,
        )
        final_regimes[majority_mask] = labels_cos + 1

        data_norms = np.linalg.norm(data_array, axis=1, keepdims=True)
        normalized_data = data_array / (data_norms + 1e-10)
        cosine_dists_full_eff = 1 - np.dot(normalized_data, centroids_cos_eff.T)
        cosine_probs_eff = compute_cluster_probabilities(cosine_dists_full_eff)
        cosine_probs_full[:, :k_eff] = cosine_probs_eff

        centroids_cos = np.zeros((r, centroids_cos_eff.shape[1]), dtype=float)
        centroids_cos[:k_eff] = centroids_cos_eff

    regime_zero_probs = l2_probs[:, index_least_freq]
    regime_probs = combine_regime_probabilities(regime_zero_probs, cosine_probs_full)

    return (
        final_regimes,
        regime_probs,
        pred_l2,
        pred_cos,
        model_l2.cluster_centers_,
        centroids_cos,
        index_least_freq,
    )


def KMeansCosine_multi(data, k, epsilon=1e-4, n_init=10, random_state=None):
    rng = np.random.default_rng(random_state)

    best_inertia = np.inf
    best_result = None

    for _ in range(n_init):
        seed = rng.integers(0, 1_000_000_000)
        labels, centroids, dists = KMeansCosine(data, k, epsilon=epsilon, random_state=seed)
        inertia = np.sum(np.min(dists, axis=1))

        if inertia < best_inertia:
            best_inertia = inertia
            best_result = (labels, centroids, dists)

    return best_result


def KMeansCosine(data: np.ndarray, k: int, epsilon: float, random_state=None):
    rng = np.random.default_rng(random_state)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / (norms + 1e-10)

    n, _ = normalized_data.shape

    indices = rng.choice(n, k, replace=False)
    centroids = normalized_data[indices]

    converged = False
    labels = np.zeros(n, dtype=int)

    while not converged:
        dists = 1 - np.dot(normalized_data, centroids.T)
        labels = np.argmin(dists, axis=1)

        new_centroids = np.zeros_like(centroids)

        for i in range(k):
            points_in_cluster = normalized_data[labels == i]

            if len(points_in_cluster) == 0:
                random_point = normalized_data[rng.integers(0, n)]
                new_centroids[i, :] = random_point / (np.linalg.norm(random_point) + 1e-10)
            else:
                mean_vec = np.mean(points_in_cluster, axis=0)
                new_centroids[i, :] = mean_vec / (np.linalg.norm(mean_vec) + 1e-10)

        if np.allclose(new_centroids, centroids, atol=epsilon):
            converged = True
        else:
            centroids = new_centroids

    final_dists = 1 - np.dot(normalized_data, centroids.T)
    return labels, centroids, final_dists


def plot_kmeans_regimes(data, regimes, recessions=None, output_path=None, show=False):
    df_plot = data.copy()
    df_plot = df_plot.iloc[len(df_plot) - len(regimes):]
    df_plot["regime_label"] = regimes

    fig, ax = plt.subplots(figsize=(15, 5))
    unique_regimes = sorted(df_plot["regime_label"].unique())

    for regime in unique_regimes:
        mask = df_plot["regime_label"] == regime
        ax.scatter(
            df_plot.loc[mask].index,
            df_plot.loc[mask, "regime_label"],
            label=f"Regime {regime}",
            alpha=0.3,
        )
    if recessions:
        for start, end in recessions:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), color="grey", alpha=0.3, zorder=0)

    ax.set_ylabel("K-Means Regimes")
    ax.set_xlabel("Date")
    ax.set_yticks(unique_regimes)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=output_path.suffix.lstrip('.') or 'svg', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
