import pandas as pd
import numpy as np
from scipy.special import expit
from sklearn.ensemble import IsolationForest

try:
    from .modified_Kmeans import KMeansCosine, plot_kmeans_regimes
    from .UMAP import fit_umap
    from .regime_hmm import fit_gmm_hmm
    from ..pipeline.config import CONTAMINATION_RATE
except ImportError:  # pragma: no cover - supports direct script execution
    from modified_Kmeans import KMeansCosine, plot_kmeans_regimes
    from UMAP import fit_umap
    from regime_hmm import fit_gmm_hmm
    from ..pipeline.config import CONTAMINATION_RATE

def Isolation_KMeans(data: pd.DataFrame, r: int = 5):
    model_Isolation = IsolationForest(n_estimators=100, bootstrap=True)
    pred_isolation = model_Isolation.fit_predict(data)

    mask_1 = pred_isolation == 1
    count_1 = np.sum(mask_1)
    count_0 = len(data) - count_1

    if count_0 > count_1:
        index_least_freq = 1
        majority_mask = ~mask_1
    else:
        index_least_freq = 0
        majority_mask = mask_1

    final_regimes = np.zeros(len(data), dtype=int)
    data_to_split = np.array(data[majority_mask])

    labels_cos, centroids_cos, pred_cos = KMeansCosine(
        data_to_split, k=r, epsilon=1e-4)
    final_regimes[majority_mask] = labels_cos + 1

    return (
        final_regimes,
        pred_isolation,
        pred_cos,
        centroids_cos,
        index_least_freq,
    )


def Isolation(X: pd.DataFrame, n_estimators: int, contamination="auto"):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        bootstrap=True,
    )
    pred_iso = model.fit_predict(X)
    regime0 = X[pred_iso == -1]
    normal = X[pred_iso == 1]
    r = regime0.shape[0] / X.shape[0]
    return regime0, normal, r


def _soft_anomaly_probs(iso_model: IsolationForest, data_values: np.ndarray, iso_score_scale: float) -> np.ndarray:
    """Convert IsolationForest decision scores to P(anomaly) via scaled sigmoid.

    decision_function > 0  → inlier (normal), < 0 → outlier (anomaly).
    We standardise by std-dev so the scale is dataset-agnostic, then apply
    expit(-z * iso_score_scale) which maps:
      z ≫ 0  →  P(anomaly) ≈ 0   (clearly normal)
      z = 0  →  P(anomaly) = 0.5  (at the decision boundary)
      z ≪ 0  →  P(anomaly) ≈ 1   (clearly anomalous)
    """
    scores = iso_model.decision_function(data_values)
    std = scores.std()
    z = scores / std if std > 1e-10 else scores
    return expit(-z * iso_score_scale)


def Isolation_UMAP_HMM(
    data: pd.DataFrame,
    r: int = 5,
    n_estimators: int = 100,
    contamination=CONTAMINATION_RATE,
    umap_components: int = 4,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    umap_metric: str = "cosine",
    umap_epochs: int = 200,
    hmm_covariance_type: str = "diag",
    hmm_n_iter: int = 1000,
    hmm_tol: float = 1e-2,
    iso_score_scale: float = 5.0,
    prob_mode: str = "soft",
    random_state: int = 42,
):
    """Regime detection via Isolation Forest → UMAP → Gaussian HMM.

    Regime 0 is the anomalous / crisis cluster identified by Isolation Forest.
    Regimes 1..r are the HMM states fitted in UMAP space.

    prob_mode="hard"
        Binary anomaly split.  Anomalies: p = [1, 0, …, 0].
        HMM is fitted only on normal points; posteriors fill p[1..r].

    prob_mode="soft"
        Soft anomaly score from the Isolation Forest decision function
        (scaled sigmoid, see _soft_anomaly_probs).  HMM is fitted on ALL
        UMAP-reduced points.  Final probability vector:
            p[0]   = p_anomaly
            p[1:r+1] = (1 - p_anomaly) * HMM_posterior
        This is the proper two-level decomposition:
        P(regime k | x) = P(normal | x) · P(HMM state k | x, normal)

    Returns
    -------
    final_regimes   : np.ndarray[int], shape (n,)   — 0=anomaly, 1..r=HMM
    regime_probs    : np.ndarray[float], shape (n, r+1)
    pred_isolation  : np.ndarray[int], shape (n,)   — +1 / -1
    umap_reduced_df : pd.DataFrame, shape (n, umap_components)
    hmm_states_full : np.ndarray[int], shape (n,)   — -1 for anomalies (hard)
    umap_mapper     : fitted umap.UMAP object
    hmm_model       : fitted GaussianHMM object (or None on failure)
    anomaly_mask    : np.ndarray[bool], shape (n,)
    """
    n = len(data)

    # ── 1. Isolation Forest ────────────────────────────────────────────────
    iso_model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        bootstrap=True,
        random_state=random_state,
    )
    pred_isolation = iso_model.fit_predict(data.values)   # +1 normal / -1 anomaly
    anomaly_mask = pred_isolation == -1
    normal_mask = ~anomaly_mask

    # ── 2. UMAP on all data ────────────────────────────────────────────────
    n_neighbors_safe = min(umap_n_neighbors, n - 1)
    n_components_safe = min(umap_components, data.shape[1])
    umap_reduced_df, umap_mapper = fit_umap(
        data,
        n_components=n_components_safe,
        n_neighbors=n_neighbors_safe,
        min_dist=umap_min_dist,
        random_state=random_state,
        metric=umap_metric,
        epochs=umap_epochs,
    )

    final_regimes = np.zeros(n, dtype=int)
    regime_probs = np.zeros((n, r + 1), dtype=float)
    hmm_model = None
    hmm_states_full = np.full(n, -1, dtype=int)

    # ── 3a. Hard mode: HMM on normal UMAP points only ─────────────────────
    if prob_mode == "hard":
        regime_probs[anomaly_mask, 0] = 1.0

        n_normal = int(normal_mask.sum())
        r_eff = min(r, n_normal)

        if r_eff > 0:
            umap_normal = umap_reduced_df[normal_mask]
            states_series, state_probs_df, hmm_model = fit_gmm_hmm(
                umap_normal,
                n_components=r_eff,
                covariance_type=hmm_covariance_type,
                n_iter=hmm_n_iter,
                tol=hmm_tol,
                plot_convergence=False,
                random_state=random_state,
            )
            if states_series is not None:
                hmm_labels = states_series.values          # 0-indexed, shape (n_normal,)
                final_regimes[normal_mask] = hmm_labels + 1
                hmm_states_full[normal_mask] = hmm_labels
                regime_probs[normal_mask, 1:r_eff + 1] = state_probs_df.values
            else:
                # Fallback: equal weight across normal regimes
                regime_probs[normal_mask, 1:] = 1.0 / r
                final_regimes[normal_mask] = 1

    # ── 3b. Soft mode: HMM on all UMAP points, blend with anomaly score ───
    else:
        states_series_all, state_probs_all_df, hmm_model = fit_gmm_hmm(
            umap_reduced_df,
            n_components=r,
            covariance_type=hmm_covariance_type,
            n_iter=hmm_n_iter,
            tol=hmm_tol,
            plot_convergence=False,
            random_state=random_state,
        )
        if states_series_all is not None:
            hmm_posterior = state_probs_all_df.values      # shape (n, r)
            hmm_states_full = states_series_all.values
        else:
            hmm_posterior = np.full((n, r), 1.0 / r)
            hmm_states_full = np.zeros(n, dtype=int)

        p_anomaly = _soft_anomaly_probs(iso_model, data.values, iso_score_scale)

        # Two-level decomposition: P(regime 0) = p_anomaly
        #                          P(regime k) = P(normal) * P(HMM state k | normal)
        regime_probs[:, 0] = p_anomaly
        regime_probs[:, 1:] = (1.0 - p_anomaly[:, None]) * hmm_posterior

        row_sums = regime_probs.sum(axis=1, keepdims=True)
        np.divide(regime_probs, row_sums, out=regime_probs, where=row_sums > 1e-12)

        final_regimes = np.argmax(regime_probs, axis=1).astype(int)

    return (
        final_regimes,
        regime_probs,
        pred_isolation,
        umap_reduced_df,
        hmm_states_full,
        umap_mapper,
        hmm_model,
        anomaly_mask,
    )
