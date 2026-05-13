from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def make_lagged_features(data: pd.DataFrame, lag: int) -> pd.DataFrame:
    """
    Build [X_t, X_{t-1}, ..., X_{t-lag}] and drop rows with missing lag history.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    lagged_blocks = []
    for l in range(0, lag + 1):
        block = data.shift(l).copy()
        if l == 0:
            block.columns = [f"{col}_lag0" for col in data.columns]
        else:
            block.columns = [f"{col}_lag{l}" for col in data.columns]
        lagged_blocks.append(block)

    lagged_df = pd.concat(lagged_blocks, axis=1)
    lagged_df = lagged_df.dropna()
    lagged_df.index.name = data.index.name
    return lagged_df


def optimal_lag_PCA(
    data: pd.DataFrame,
    lag: int,
    target_variance: float = 0.95,
    plot: bool = False,
):
    """
    Run PCA on lag-expanded features.
    Returns:
        reduced_df: PCA scores
        n_components_optimal: selected component count
        final_model: fitted PCA model
        lagged_df: lag-expanded input matrix used for PCA
    """
    lagged_df = make_lagged_features(data, lag=lag)

    model = PCA()
    model.fit(lagged_df)

    exp_var_ratio = model.explained_variance_ratio_
    cumulative_var = np.cumsum(exp_var_ratio)
    n_components_optimal = np.argmax(cumulative_var >= target_variance) + 1

    final_model = PCA(n_components=n_components_optimal)
    transformed = final_model.fit_transform(lagged_df)

    component_names = [f"PC{i + 1}" for i in range(transformed.shape[1])]
    reduced_df = pd.DataFrame(
        transformed,
        columns=component_names,
        index=lagged_df.index,
    )
    reduced_df.index.name = lagged_df.index.name

    return reduced_df, n_components_optimal, final_model, lagged_df