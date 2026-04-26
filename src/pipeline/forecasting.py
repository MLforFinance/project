from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .config import BLACK_LITTERMAN_TAU


def regime_weights(probs_next: np.ndarray, forecast_mode: str) -> np.ndarray:
    probs = np.asarray(probs_next, dtype=float)
    if probs.ndim != 1:
        raise ValueError("probs_next must be a one-dimensional array.")

    total = probs.sum()
    if total <= 0:
        probs = np.ones_like(probs, dtype=float) / len(probs)
    else:
        probs = probs / total

    if forecast_mode == "soft":
        return probs
    if forecast_mode == "hard":
        hard = np.zeros_like(probs, dtype=float)
        hard[int(np.argmax(probs))] = 1.0
        return hard
    raise ValueError("forecast_mode must be one of: 'hard', 'soft'.")


def _regime_weighted_mean(returns_df: pd.DataFrame, regimes: pd.Series, probs_next: np.ndarray, forecast_mode: str) -> pd.Series:
    probs = regime_weights(probs_next, forecast_mode)
    weighted_sum = pd.Series(0.0, index=returns_df.columns, dtype=float)
    total_weight = 0.0

    for regime, weight in enumerate(probs):
        if weight <= 0:
            continue
        subset = returns_df.loc[regimes == regime]
        if subset.empty:
            continue
        weighted_sum = weighted_sum.add(subset.mean() * weight, fill_value=0.0)
        total_weight += float(weight)

    if total_weight <= 0:
        return returns_df.mean()
    return weighted_sum / total_weight


def _regime_weighted_std(returns_df: pd.DataFrame, regimes: pd.Series, probs_next: np.ndarray, forecast_mode: str) -> pd.Series:
    probs = regime_weights(probs_next, forecast_mode)
    weighted_sum = pd.Series(0.0, index=returns_df.columns, dtype=float)
    total_weight = 0.0

    for regime, weight in enumerate(probs):
        if weight <= 0:
            continue
        subset = returns_df.loc[regimes == regime]
        if subset.empty:
            continue
        weighted_sum = weighted_sum.add(subset.std() * weight, fill_value=0.0)
        total_weight += float(weight)

    if total_weight <= 0:
        return returns_df.std()
    return weighted_sum / total_weight


def forecast_naive_sharpe(returns_df: pd.DataFrame, regimes: pd.Series, probs_next: np.ndarray, forecast_mode: str = "soft") -> pd.Series:
    mu = _regime_weighted_mean(returns_df, regimes, probs_next, forecast_mode)
    sigma = _regime_weighted_std(returns_df, regimes, probs_next, forecast_mode).replace(0, 1e-8)
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


def predict_ridge(
    models: dict[int, Ridge],
    X_t: pd.Series | np.ndarray,
    probs_next: np.ndarray,
    n_regimes: int,
    forecast_mode: str = "soft",
) -> np.ndarray:
    X_values = np.asarray(X_t).reshape(1, -1)

    if -1 in models:
        return np.asarray(models[-1].predict(X_values)[0])

    weights = regime_weights(probs_next, forecast_mode)
    preds = None
    total_weight = 0.0

    for regime, model in models.items():
        if regime < 0 or regime >= n_regimes:
            continue
        weight = float(weights[regime])
        if weight <= 0:
            continue
        pred_regime = model.predict(X_values)[0]
        weighted = weight * pred_regime
        preds = weighted if preds is None else preds + weighted
        total_weight += weight

    if preds is None or total_weight <= 0:
        available_regimes = [regime for regime in models if 0 <= regime < n_regimes]
        if not available_regimes:
            return np.zeros(X_values.shape[1], dtype=float)
        fallback_regime = max(available_regimes, key=lambda regime: float(weights[regime]))
        return np.asarray(models[fallback_regime].predict(X_values)[0])

    if forecast_mode == "soft" and total_weight < 1.0:
        preds = preds / total_weight
    return np.asarray(preds)


def forecast_black_litterman_scores(
    returns_df: pd.DataFrame,
    regimes: pd.Series,
    probs_next: np.ndarray,
    tau: float = BLACK_LITTERMAN_TAU,
    forecast_mode: str = "soft",
) -> pd.Series:
    mu_prior = returns_df.mean().to_numpy()
    sigma = returns_df.cov().to_numpy(copy=True)
    sigma += np.eye(len(mu_prior)) * 1e-6

    q_star = _regime_weighted_mean(returns_df, regimes, probs_next, forecast_mode).to_numpy()

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
