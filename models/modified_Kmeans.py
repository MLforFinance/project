import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def modified_Kmeans(data:pd.DataFrame, r = 4):
    modelL2 = KMeans(n_clusters = 2)
    predL2 = modelL2.fit_predict(data)
    A = data[predL2 == 1]
    B = data[predL2 == 0]

    data_cosine = A.copy() if A.shape[0] < B.shape[0] else B.copy()
    data_cosine = normalize(data_cosine)

    modelCosine = KMeans(n_clusters=r)
    predCos = modelCosine.fit_predict(data_cosine)

    return predL2, predCos, modelL2.cluster_centers_, modelCosine.cluster_centers_


def cosine_distance(x, y):
    return np.dot(x,y) / np.sqrt(np.linalg.norm(x) * np.linalg.norm(y))

def l2_distance(x, y):
    return np.linalg.norm(x-y)


if __name__ == "__main__":
    data = pd.read_csv("data/2026-02-MD_reduced.csv", index_col = 0)
    
    predL2, predCos, clustersL2, clustersCos = modified_Kmeans(data)
    print(predL2, predCos)