import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader



def get_loaders(path_csv: str, test_p: float = 0.2):
    
    data = pd.read_csv(path_csv, index_col=0)
    train_set, test_set = train_test_split(data, test_size=test_p)

    train_tensor = torch.tensor(train_set.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_set.values, dtype=torch.float32)

    return (
        DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True), 
        DataLoader(TensorDataset(test_tensor), batch_size=64, shuffle=False)
    )


def get_data(csv_name:str):
    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR.parent.parent / "data" / csv_name
    return pd.read_csv(data_path, index_col = 0)


def condition_number(X:pd.DataFrame):
    u, s, v = np.linalg.svd(X)
    return s[0] / s[-1]


def feature_level_col(X:pd.DataFrame, n:int = 20):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) 
                    for i in range(len(X.columns))]

    return vif_data.sort_values(by="VIF", ascending=False).head(n), (vif_data["VIF"] > 10).count()



def plot_regimes(data, regimes, recessions=None, output_path=None, show=False):
    df_plot = data.copy()
    df_plot = df_plot.iloc[len(df_plot) - len(regimes):]
    df_plot["regime_label"] = regimes
    df_plot["index_plot"] = np.arange(0, df_plot.shape[0], step = 1)

    fig, ax = plt.subplots(figsize=(15, 5))
    unique_regimes = sorted(df_plot["regime_label"].unique())

    for r in unique_regimes:
        mask = df_plot["regime_label"] == r
        ax.scatter(
            df_plot.loc[mask, "index_plot"],
            df_plot.loc[mask, "regime_label"],
            label=f"Regime {r}",
            alpha=0.3,
        )
    if recessions:
        for start, end in recessions:
            ax.axvspan(pd.to_datetime(start), pd.to_datetime(end),
                       color='grey', alpha=0.3, zorder=0)

    ax.set_ylabel('Regimes')
    ax.set_xlabel('Date')
    ax.set_yticks(unique_regimes)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format=output_path.suffix.lstrip(
            '.') or 'svg', bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


