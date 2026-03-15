import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def modified_KMeans(data: pd.DataFrame, r: int = 5):
    model_l2 = KMeans(n_clusters=2, n_init='auto')
    pred_l2 = model_l2.fit_predict(data)

    mask_1 = (pred_l2 == 1)
    count_1 = np.sum(mask_1)
    count_0 = len(data) - count_1

    if count_0 > count_1:
        minority_mask = mask_1
        majority_mask = ~mask_1
    else:
        minority_mask = ~mask_1
        majority_mask = mask_1

    final_regimes = np.zeros(len(data), dtype=int)
    
    data_to_split = data[majority_mask]
    data_cosine_scaled = normalize(data_to_split)

    model_cosine = KMeans(n_clusters=r, n_init='auto')
    pred_cos = model_cosine.fit_predict(data_cosine_scaled)

    final_regimes[majority_mask] = pred_cos + 1

    return (
        final_regimes, 
        pred_l2, 
        pred_cos, 
        model_l2.cluster_centers_, 
        model_cosine.cluster_centers_
    )

def plot_kmeans_regimes(data, regimes, recessions=None):
    df_plot = data.copy()
    df_plot["sasdate"] = pd.to_datetime(df_plot["sasdate"].iloc[3:], format="%d/%m/%Y")
    
    df_plot = df_plot.iloc[len(df_plot) - len(regimes):]
    
    df_plot["regime_label"] = regimes
    
    fig, ax = plt.subplots(figsize=(15, 5))

    unique_regimes = sorted(df_plot["regime_label"].unique())

    for i, r in enumerate(unique_regimes):
        mask = (df_plot["regime_label"] == r)
        ax.scatter(df_plot.loc[mask, "sasdate"], 
                   df_plot.loc[mask, "regime_label"], 
                   label=f"Regime {r}", 
                   s=30, 
                   alpha=0.7)
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


def cosine_distance(x, y):
    return np.dot(x,y) / np.sqrt(np.linalg.norm(x) * np.linalg.norm(y))

def l2_distance(x, y):
    return np.linalg.norm(x-y)


if __name__ == "__main__":
    reduced_data = pd.read_csv("data/2026-02-MD_reduced.csv", index_col = 0)
    raw_data = pd.read_csv("data/2026-02-MD.csv")
    
    regimes_pred, predL2, predCos, clustersL2, clustersCos = modified_KMeans(reduced_data)

    plot_kmeans_regimes(raw_data, regimes_pred)


    