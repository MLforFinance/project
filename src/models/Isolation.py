import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

<<<<<<< Updated upstream:src/models/Isolation.py
try:
    from .modified_Kmeans import KMeansCosine, plot_kmeans_regimes
except ImportError:  # pragma: no cover - supports direct script execution
    from modified_Kmeans import KMeansCosine, plot_kmeans_regimes
=======
from .modified_Kmeans import KMeansCosine, plot_kmeans_regimes

>>>>>>> Stashed changes:models/Isolation.py


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
