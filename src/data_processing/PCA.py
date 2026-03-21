from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def optimal_PCA(
    data: pd.DataFrame,
    target_variance: float = 0.95,
    plot: bool = True,
) -> tuple[pd.DataFrame, int, PCA]:
    model = PCA()
    model.fit(data)

    exp_var_ratio = model.explained_variance_ratio_
    cumulative_var = np.cumsum(exp_var_ratio)
    n_components_optimal = np.argmax(cumulative_var >= target_variance) + 1

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(cumulative_var) + 1),
                cumulative_var, linestyle="--")
        ax.axhline(
            y=target_variance,
            color="r",
            linestyle="--",
            label=f"Target {target_variance * 100:.0f}%",
        )
        ax.axvline(
            x=n_components_optimal,
            color="r",
            linestyle="--",
            label=f"Optimal: {n_components_optimal}",
        )
        ax.set_xlabel("Components")
        ax.set_ylabel("Cumulative explained variance")
        ax.legend()
        plt.tight_layout()
        plt.show()

    final_model = PCA(n_components=n_components_optimal)
    transformed = final_model.fit_transform(data)
    component_names = [f"PC{i + 1}" for i in range(transformed.shape[1])]
    reduced_df = pd.DataFrame(
        transformed, columns=component_names, index=data.index)
    reduced_df.index.name = data.index.name
    return reduced_df, n_components_optimal, final_model


def run_pca(
    input_path: str | Path,
    output_path: str | Path | None = None,
    target_variance: float = 0.95,
    plot: bool = True,
) -> tuple[pd.DataFrame, int, PCA]:
    data = pd.read_csv(input_path, index_col=0, parse_dates=True)
    reduced_df, n_components, model = optimal_PCA(
        data,
        target_variance=target_variance,
        plot=plot,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_df.to_csv(output_path)

    return reduced_df, n_components, model
