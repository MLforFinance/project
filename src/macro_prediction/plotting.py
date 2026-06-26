"""Plot helpers for macro regime outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_regime_transitions(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
    regime_column: str = "predicted_regime",
) -> None:
    """Plot predicted regime number over time."""

    if regime_column not in predictions.columns:
        raise ValueError(f"Missing regime column: {regime_column}")

    regimes = predictions[regime_column].astype(int)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.step(regimes.index, regimes, where="post", linewidth=1.8, color="#1f77b4")
    ax.scatter(regimes.index, regimes, s=12, color="#1f77b4", alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Regime number")
    ax.set_yticks(sorted(regimes.unique()))
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(True, axis="x", alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
