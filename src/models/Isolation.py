from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


RANDOM_SEED = 42
KMEANS_N_INIT = 10
FUZZY_M = 2.0
EPS = 1e-12


def isolation_rank_probabilities(
    model: IsolationForest,
    data: pd.DataFrame,
    full_abnormal_percentile: float = 0.95,
    zero_abnormal_percentile: float = 0.50,
) -> np.ndarray:
    """Convert Isolation Forest anomaly scores into soft normal/abnormal probabilities.

    Isolation Forest itself returns anomaly scores, not true probabilities. This
    function uses the rank-percentile approach requested for the project:

    - observations with the largest anomaly scores are the most abnormal;
    - observations at or above `full_abnormal_percentile` get P(abnormal)=1;
    - observations at or below `zero_abnormal_percentile` get P(abnormal)=0;
    - observations between those cutoffs are linearly scaled from 0 to 1.

    The output has two columns:
        column 0 = P(normal)
        column 1 = P(abnormal)
    """
    if len(data) == 0:
        return np.empty((0, 2), dtype=float)

    # sklearn score_samples: larger = more normal, smaller = more abnormal.
    # Therefore multiply by -1 so larger = more abnormal.
    anomaly_score = -model.score_samples(data)
    ranks = pd.Series(anomaly_score).rank(method="average", pct=True).to_numpy()

    if not 0.0 <= zero_abnormal_percentile < full_abnormal_percentile <= 1.0:
        raise ValueError("Require 0 <= zero_abnormal_percentile < full_abnormal_percentile <= 1.")

    # Project rule:
    # - top 5% most abnormal => percentile >= 0.95 => P(abnormal)=1
    # - bottom 50% by abnormality => percentile <= 0.50 => P(abnormal)=0
    # - between 50% and 95% => linear ramp from 0 to 1
    abnormal_prob = (ranks - zero_abnormal_percentile) / (full_abnormal_percentile - zero_abnormal_percentile)
    abnormal_prob = np.clip(abnormal_prob, 0.0, 1.0)
    normal_prob = 1.0 - abnormal_prob

    return np.column_stack([normal_prob, abnormal_prob])


def fuzzy_memberships_from_distances(distances: np.ndarray, m: float = FUZZY_M) -> np.ndarray:
    """Convert distances to cluster centers into fuzzy C-means memberships.

    For m=2, the membership of point i in cluster k is:
        u_ik = 1 / sum_j (d_ik / d_ij)^2

    Rows sum to 1. If a point is exactly on a centroid, it receives 100% weight
    on that centroid.
    """
    distances = np.asarray(distances, dtype=float)
    if distances.ndim != 2:
        raise ValueError("distances must be a two-dimensional array")
    if distances.shape[0] == 0:
        return np.empty_like(distances)

    n_obs, n_clusters = distances.shape
    memberships = np.zeros((n_obs, n_clusters), dtype=float)
    power = 2.0 / (m - 1.0)

    for i in range(n_obs):
        d = np.maximum(distances[i], 0.0)
        zero_mask = d <= EPS

        if zero_mask.any():
            memberships[i, zero_mask] = 1.0 / zero_mask.sum()
            continue

        ratios = (d[:, None] / d[None, :]) ** power
        memberships[i] = 1.0 / ratios.sum(axis=1)

    row_sums = memberships.sum(axis=1, keepdims=True)
    uniform = np.full_like(memberships, 1.0 / max(n_clusters, 1))
    return np.divide(memberships, row_sums, out=uniform, where=row_sums > 0)


def Isolation_Euclidean_KMeans(
    data: pd.DataFrame,
    r: int = 5,
    random_state: int = RANDOM_SEED,
):
    """Detect abnormal observations and cluster normal observations with soft probabilities.

    Regime 0 is the abnormal/isolation regime.
    Regimes 1..r are Kernel-PCA/KMeans regimes.

    Hard labels are still returned for plotting and transition counting, but this
    function now also returns a probability matrix:
        regime_probs[:, 0]      = P(abnormal)
        regime_probs[:, 1:r+1]  = P(normal) * fuzzy cluster membership
    """
    model_isolation = IsolationForest(
        n_estimators=100,
        bootstrap=True,
        random_state=random_state,
    )

    pred_isolation = model_isolation.fit_predict(data)
    isolation_probs = isolation_rank_probabilities(model_isolation, data)
    p_normal = isolation_probs[:, 0]
    p_abnormal = isolation_probs[:, 1]

    abnormal_mask = pred_isolation == -1
    normal_mask = ~abnormal_mask

    # Fallback for very small or degenerate samples.
    if normal_mask.sum() < 2:
        normal_mask = np.ones(len(data), dtype=bool)
        abnormal_mask = ~normal_mask

    final_regimes = np.zeros(len(data), dtype=int)
    data_to_split = data.loc[normal_mask].copy()

    k_eff = min(r, len(data_to_split))
    if k_eff <= 0:
        regime_probs = np.zeros((len(data), r + 1), dtype=float)
        regime_probs[:, 0] = 1.0
        return (
            final_regimes,
            pred_isolation,
            np.array([], dtype=int),
            np.empty((0, data.shape[1])),
            0,
            regime_probs,
        )

    model_kmeans = KMeans(
        n_clusters=k_eff,
        random_state=random_state,
        n_init=KMEANS_N_INIT,
        tol=1e-5,
    )

    labels_kmeans = model_kmeans.fit_predict(data_to_split)
    final_regimes[normal_mask] = labels_kmeans + 1

    # Distances from every point in Kernel PCA space to every normal-regime centroid.
    distances_full_eff = model_kmeans.transform(data)
    fuzzy_eff = fuzzy_memberships_from_distances(distances_full_eff, m=FUZZY_M)

    fuzzy_full = np.zeros((len(data), r), dtype=float)
    fuzzy_full[:, :k_eff] = fuzzy_eff

    # Combine Isolation Forest probability with fuzzy cluster probability.
    regime_probs = np.zeros((len(data), r + 1), dtype=float)
    regime_probs[:, 0] = p_abnormal
    regime_probs[:, 1:] = p_normal[:, None] * fuzzy_full

    row_sums = regime_probs.sum(axis=1, keepdims=True)
    uniform = np.full_like(regime_probs, 1.0 / (r + 1))
    regime_probs = np.divide(regime_probs, row_sums, out=uniform, where=row_sums > 0)

    index_least_freq = 0  # regime 0 is now explicitly the abnormal regime

    centers_full = np.zeros((r, data.shape[1]), dtype=float)
    centers_full[:k_eff] = model_kmeans.cluster_centers_

    return (
        final_regimes,
        pred_isolation,
        labels_kmeans,
        centers_full,
        index_least_freq,
        regime_probs,
    )
