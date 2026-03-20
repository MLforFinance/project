import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def modified_KMeans(data: pd.DataFrame, r: int = 5):
    model_l2 = KMeans(n_clusters=2, tol = 10**-5)
    pred_l2 = model_l2.fit_transform(data)

    mask_1 = (np.argmin(pred_l2, axis=1) == 1)
    count_1 = np.sum(mask_1)
    count_0 = len(data) - count_1

    if count_0 > count_1:
        index_least_freq = 1
        minority_mask = mask_1
        majority_mask = ~mask_1
    else:
        index_least_freq = 0
        minority_mask = ~mask_1
        majority_mask = mask_1

    final_regimes = np.zeros(len(data), dtype=int)
    data_to_split = np.array(data[majority_mask])

    # Custom Kmeans
    labels_cos, centroids_cos, pred_cos = KMeansCosine(data_to_split, k=r, epsilon=1e-4)

    final_regimes[majority_mask] = labels_cos + 1
    return (
        final_regimes, 
        pred_l2, 
        pred_cos, 
        model_l2.cluster_centers_, 
        centroids_cos,
        index_least_freq
    )


def KMeansCosine(data: np.ndarray, k: int, epsilon: float = 1e-4):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    normalized_data = data / (norms + 1e-10)

    n, d = normalized_data.shape
    
    centroids = np.random.uniform(np.min(normalized_data, axis=0), 
                                  np.max(normalized_data, axis=0), 
                                  (k, d))
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)

    converged = False
    labels = np.zeros(n, dtype=int)

    while not converged:
        dists = 1 - np.dot(normalized_data, centroids.T)
        labels = np.argmin(dists, axis=1)

        newCentroids = np.zeros_like(centroids)

        # Centroid update
        for i in range(k):
            points_in_cluster = normalized_data[labels == i]
            
            if len(points_in_cluster) == 0:
                random_point = normalized_data[np.random.randint(0, n)]
                newCentroids[i, :] = random_point / (np.linalg.norm(random_point) + 1e-10)
            else:
                mean_vec = np.mean(points_in_cluster, axis=0)
                newCentroids[i, :] = mean_vec / (np.linalg.norm(mean_vec) + 1e-10)

        if (np.abs(newCentroids - centroids) < epsilon).all():
            converged = True
        else:
            centroids = newCentroids

    final_dists = 1 - np.dot(normalized_data, centroids.T)
    return labels, centroids, final_dists


def plot_kmeans_regimes(data, regimes, recessions=None):
    df_plot = data.copy()
    
    df_plot = df_plot.iloc[len(df_plot) - len(regimes):]
    
    df_plot["regime_label"] = regimes
    
    fig, ax = plt.subplots(figsize=(15, 5))

    unique_regimes = sorted(df_plot["regime_label"].unique())

    for i, r in enumerate(unique_regimes):
        mask = (df_plot["regime_label"] == r)
        ax.scatter(df_plot.loc[mask].index, 
                   df_plot.loc[mask, "regime_label"], 
                   label=f"Regime {r}",
                   alpha=0.3)
    if recessions:
        for start, end in recessions:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                       color='grey', alpha=0.3, zorder=0)

    ax.set_ylabel('K-Means Regimes')
    ax.set_xlabel('Date')
    ax.set_yticks(unique_regimes)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    plt.show()

