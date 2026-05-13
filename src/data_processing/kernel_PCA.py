from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import KernelPCA


def optimal_kernel_PCA(
    data: pd.DataFrame,
    kernel: str,
    n_components: int = 6,
    gamma: float | None = None,
    degree: int = 3,
    coef0: float = 1.0,
):
    """
    Run Kernel PCA on processed data.

    Parameters
    ----------
    data : pd.DataFrame
        Input processed data.
    kernel : str
        One of: "rbf", "cosine", "poly"
    n_components : int
        Number of kernel principal components to keep.
    gamma : float | None
        Kernel coefficient for rbf/poly.
        If None, sklearn chooses its default.
    degree : int
        Degree for polynomial kernel.
    coef0 : float
        Independent term for polynomial kernel.

    Returns
    -------
    reduced_df : pd.DataFrame
        Kernel PCA scores with columns PC1, PC2, ...
    n_components : int
        Number of retained components
    model : KernelPCA
        Fitted KernelPCA model
    """
    model = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        coef0=coef0,
        fit_inverse_transform=False,
        eigen_solver="auto",
    )

    transformed = model.fit_transform(data)

    component_names = [f"PC{i + 1}" for i in range(transformed.shape[1])]
    reduced_df = pd.DataFrame(
        transformed,
        columns=component_names,
        index=data.index,
    )
    reduced_df.index.name = data.index.name

    return reduced_df, n_components, model