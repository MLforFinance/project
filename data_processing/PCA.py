import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def optimal_PCA(data: pd.DataFrame, target_variance=0.95):
    model = PCA() 
    model.fit(data)
    
    sv = model.singular_values_
    exp_var_ratio = (sv**2) / np.sum(sv**2)
    cumulative_var = np.cumsum(exp_var_ratio)

    n_components_optimal = np.argmax(cumulative_var >= target_variance) + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumulative_var) + 1), cumulative_var, linestyle='--')
    
    ax.axhline(y=target_variance, color = 'r', linestyle = '--',label=f't {target_variance*100}%')
    ax.axvline(x=n_components_optimal, color = 'r', linestyle = '--', label=f'Optimal: {n_components_optimal}')
    plt.xlabel("Components")
    plt.ylabel("Cumulative explained var")
    ax.legend()
    plt.show()

    final_model = PCA(n_components=n_components_optimal)
    return final_model.fit_transform(data), n_components_optimal



if __name__ == "__main__":
    data = pd.read_csv("data/2026-02-MD_processed.csv", index_col=0)
    data, k = optimal_PCA(data)