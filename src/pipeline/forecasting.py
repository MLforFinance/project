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


def _normalize_sample_weights(sample_weights, index: pd.Index) -> pd.Series:
    """Return non-negative sample weights aligned to index and normalized to sum to 1."""
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)

    if sample_weights is None:
        return pd.Series(1.0 / len(index), index=index, dtype=float)

    if isinstance(sample_weights, pd.Series):
        weights = sample_weights.reindex(index).astype(float)
    else:
        weights = pd.Series(np.asarray(sample_weights, dtype=float), index=index)

    weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = float(weights.sum())

    if total <= 0:
        return pd.Series(1.0 / len(index), index=index, dtype=float)

    return weights / total


def _weighted_mean(df: pd.DataFrame, sample_weights=None) -> pd.Series:
    weights = _normalize_sample_weights(sample_weights, df.index)
    return df.mul(weights, axis=0).sum(axis=0)


def _weighted_cov(df: pd.DataFrame, sample_weights=None) -> pd.DataFrame:
    """Weighted covariance matrix using normalized weights.

    This is the population-style weighted covariance, which is usually better for
    exponentially decayed estimates than the ordinary equal-weight sample covariance.
    """
    weights = _normalize_sample_weights(sample_weights, df.index)
    mu = _weighted_mean(df, weights)
    centered = df - mu
    cov = centered.mul(weights, axis=0).T @ centered
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def _weighted_std(df: pd.DataFrame, sample_weights=None) -> pd.Series:
    cov = _weighted_cov(df, sample_weights)
    variances = np.diag(cov.to_numpy(copy=True))
    variances = np.maximum(variances, 0.0)
    return pd.Series(np.sqrt(variances), index=df.columns)


def _regime_weighted_mean(
    returns_df: pd.DataFrame,
    regimes: pd.Series,
    probs_next: np.ndarray,
    forecast_mode: str,
    sample_weights=None,
) -> pd.Series:
    probs = regime_weights(probs_next, forecast_mode)
    full_weights = _normalize_sample_weights(sample_weights, returns_df.index)
    weighted_sum = pd.Series(0.0, index=returns_df.columns, dtype=float)
    total_weight = 0.0

    for regime, weight in enumerate(probs):
        if weight <= 0:
            continue
        subset = returns_df.loc[regimes == regime]
        if subset.empty:
            continue
        subset_weights = full_weights.loc[subset.index]
        weighted_sum = weighted_sum.add(_weighted_mean(subset, subset_weights) * weight, fill_value=0.0)
        total_weight += float(weight)

    if total_weight <= 0:
        return _weighted_mean(returns_df, full_weights)
    return weighted_sum / total_weight


def _regime_weighted_std(
    returns_df: pd.DataFrame,
    regimes: pd.Series,
    probs_next: np.ndarray,
    forecast_mode: str,
    sample_weights=None,
) -> pd.Series:
    probs = regime_weights(probs_next, forecast_mode)
    full_weights = _normalize_sample_weights(sample_weights, returns_df.index)
    weighted_sum = pd.Series(0.0, index=returns_df.columns, dtype=float)
    total_weight = 0.0

    for regime, weight in enumerate(probs):
        if weight <= 0:
            continue
        subset = returns_df.loc[regimes == regime]
        if subset.empty:
            continue
        subset_weights = full_weights.loc[subset.index]
        weighted_sum = weighted_sum.add(_weighted_std(subset, subset_weights) * weight, fill_value=0.0)
        total_weight += float(weight)

    if total_weight <= 0:
        return _weighted_std(returns_df, full_weights)
    return weighted_sum / total_weight


def forecast_naive_sharpe(
    returns_df: pd.DataFrame,
    regimes: pd.Series,
    probs_next: np.ndarray,
    forecast_mode: str = "soft",
    sample_weights=None,
) -> pd.Series:
    mu = _regime_weighted_mean(returns_df, regimes, probs_next, forecast_mode, sample_weights=sample_weights)
    sigma = _regime_weighted_std(returns_df, regimes, probs_next, forecast_mode, sample_weights=sample_weights).replace(0, 1e-8)
    return mu / sigma


def train_ridge_models(
    X: pd.DataFrame | np.ndarray,
    Y: pd.DataFrame | np.ndarray,
    regimes: pd.Series,
    n_regimes: int,
    alpha: float = 1.0,
    sample_weights=None,
) -> dict[int, Ridge]:
    X_values = np.asarray(X)
    Y_values = np.asarray(Y)
    regime_values = np.asarray(regimes)
    sample_weights_values = None if sample_weights is None else np.asarray(sample_weights, dtype=float)
    models: dict[int, Ridge] = {}

    for regime in range(n_regimes):
        mask = regime_values == regime
        if mask.sum() < 5:
            continue
        model = Ridge(alpha=alpha)
        if sample_weights_values is None:
            model.fit(X_values[mask], Y_values[mask])
        else:
            model.fit(X_values[mask], Y_values[mask], sample_weight=sample_weights_values[mask])
        models[regime] = model

    if not models:
        fallback = Ridge(alpha=alpha)
        if sample_weights_values is None:
            fallback.fit(X_values, Y_values)
        else:
            fallback.fit(X_values, Y_values, sample_weight=sample_weights_values)
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
    sample_weights=None,
) -> pd.Series:
    mu_prior = _weighted_mean(returns_df, sample_weights).to_numpy()
    sigma = _weighted_cov(returns_df, sample_weights).to_numpy(copy=True)
    sigma += np.eye(len(mu_prior)) * 1e-6

    q_star = _regime_weighted_mean(
        returns_df,
        regimes,
        probs_next,
        forecast_mode,
        sample_weights=sample_weights,
    ).to_numpy()

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega = np.diag(np.diag(tau_sigma)) + np.eye(len(mu_prior)) * 1e-6
    omega_inv = np.linalg.pinv(omega)

    posterior = np.linalg.pinv(tau_sigma_inv + omega_inv) @ (tau_sigma_inv @ mu_prior + omega_inv @ q_star)
    return pd.Series(posterior, index=returns_df.columns)


def forecast_mvo_scores(returns_df: pd.DataFrame, sample_weights=None) -> pd.Series:
    mu = _weighted_mean(returns_df, sample_weights).to_numpy()
    sigma = _weighted_cov(returns_df, sample_weights).to_numpy(copy=True) + np.eye(returns_df.shape[1]) * 1e-6
    scores = np.linalg.pinv(sigma) @ mu
    return pd.Series(scores, index=returns_df.columns)


def compute_random_regime_state(index: pd.Index, n_regimes: int, rng: np.random.Generator) -> tuple[pd.Series, pd.DataFrame]:
    random_regimes = pd.Series(rng.integers(0, n_regimes, size=len(index)), index=index, name="regime")
    probs = np.zeros((len(index), n_regimes), dtype=float)
    probs[np.arange(len(index)), random_regimes.to_numpy()] = 1.0
    probs_df = pd.DataFrame(probs, index=index, columns=[f"regime_prob_{i}" for i in range(n_regimes)])
    return random_regimes, probs_df
