from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import BLACK_LITTERMAN_TAU


def forecast_naive_sharpe(returns_df: pd.DataFrame, regimes: pd.Series, probs_next: np.ndarray) -> pd.Series:
    target_regime = int(np.argmax(probs_next))
    subset = returns_df.loc[regimes == target_regime]
    if subset.empty:
        subset = returns_df
    mu = subset.mean()
    sigma = subset.std().replace(0, 1e-8)
    return mu / sigma


def train_ridge_models(
    X: pd.DataFrame | np.ndarray,
    Y: pd.DataFrame | np.ndarray,
    regimes: pd.Series,
    n_regimes: int,
    alpha: float = 1.0,
) -> dict[int, Ridge]:
    X_values = np.asarray(X)
    Y_values = np.asarray(Y)
    regime_values = np.asarray(regimes)
    models: dict[int, Ridge] = {}

    for regime in range(n_regimes):
        mask = regime_values == regime
        if mask.sum() < 5:
            continue
        model = Ridge(alpha=alpha)
        model.fit(X_values[mask], Y_values[mask])
        models[regime] = model

    if not models:
        fallback = Ridge(alpha=alpha)
        fallback.fit(X_values, Y_values)
        models[-1] = fallback

    return models


def predict_ridge(models: dict[int, Ridge], X_t: pd.Series | np.ndarray, probs_next: np.ndarray, n_regimes: int) -> np.ndarray:
    X_values = np.asarray(X_t).reshape(1, -1)
    preds = None

    for regime, model in models.items():
        pred_regime = model.predict(X_values)[0]
        weight = 1.0 if regime == -1 else probs_next[regime]
        weighted = weight * pred_regime
        preds = weighted if preds is None else preds + weighted

    return np.asarray(preds)


def forecast_black_litterman_scores(
    returns_df: pd.DataFrame,
    regimes: pd.Series,
    probs_next: np.ndarray,
    tau: float = BLACK_LITTERMAN_TAU,
) -> pd.Series:
    mu_prior = returns_df.mean().to_numpy()
    sigma = returns_df.cov().to_numpy(copy=True)
    sigma += np.eye(len(mu_prior)) * 1e-6

    target_regime = int(np.argmax(probs_next))
    subset = returns_df.loc[regimes == target_regime]
    if subset.empty:
        subset = returns_df
    q_star = subset.mean().to_numpy()

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega = np.diag(np.diag(tau_sigma)) + np.eye(len(mu_prior)) * 1e-6
    omega_inv = np.linalg.pinv(omega)

    posterior = np.linalg.pinv(tau_sigma_inv + omega_inv) @ (tau_sigma_inv @ mu_prior + omega_inv @ q_star)
    return pd.Series(posterior, index=returns_df.columns)


def forecast_mvo_scores(returns_df: pd.DataFrame) -> pd.Series:
    mu = returns_df.mean().to_numpy()
    sigma = returns_df.cov().to_numpy(copy=True) + np.eye(returns_df.shape[1]) * 1e-6
    scores = np.linalg.pinv(sigma) @ mu
    return pd.Series(scores, index=returns_df.columns)


def compute_random_regime_state(index: pd.Index, n_regimes: int, rng: np.random.Generator) -> tuple[pd.Series, pd.DataFrame]:
    random_regimes = pd.Series(rng.integers(0, n_regimes, size=len(index)), index=index, name="regime")
    probs = np.zeros((len(index), n_regimes), dtype=float)
    probs[np.arange(len(index)), random_regimes.to_numpy()] = 1.0
    probs_df = pd.DataFrame(probs, index=index, columns=[f"regime_prob_{i}" for i in range(n_regimes)])
    return random_regimes, probs_df
