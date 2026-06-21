from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd


RANDOM_SEED = 42
KMEANS_N_INIT = 10


def Isolation_Euclidean_KMeans(
    data: pd.DataFrame,
    r: int = 5,
    random_state: int = RANDOM_SEED,
):
    model_isolation = IsolationForest(
        n_estimators=100,
        bootstrap=True,
        random_state=random_state,
    )

    pred_isolation = model_isolation.fit_predict(data)

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

    data_to_split = data.loc[majority_mask].copy()

    model_kmeans = KMeans(
        n_clusters=r,
        random_state=random_state,
        n_init=KMEANS_N_INIT,
        tol=1e-5,
    )

    labels_kmeans = model_kmeans.fit_predict(data_to_split)

    final_regimes[majority_mask] = labels_kmeans + 1

    return (
        final_regimes,
        pred_isolation,
        model_kmeans.labels_,
        model_kmeans.cluster_centers_,
        index_least_freq,
    )