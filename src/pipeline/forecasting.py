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




def _normalize_regime_probability_frame(
    regime_probabilities: pd.DataFrame | np.ndarray | None,
    index: pd.Index,
    n_regimes: int | None = None,
) -> pd.DataFrame | None:
    """Return soft regime probabilities aligned to index.

    Rows are normalized to sum to 1. This lets the forecasting step estimate
    regime-specific returns from fuzzy memberships rather than only hard labels.
    """
    if regime_probabilities is None:
        return None

    if isinstance(regime_probabilities, pd.DataFrame):
        probs = regime_probabilities.reindex(index).astype(float)
    else:
        probs = pd.DataFrame(np.asarray(regime_probabilities, dtype=float), index=index)

    if n_regimes is not None:
        probs = probs.iloc[:, :n_regimes]

    probs = probs.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    row_sums = probs.sum(axis=1)
    valid = row_sums > 0
    if (~valid).any():
        probs.loc[~valid, :] = 1.0 / probs.shape[1]
        row_sums = probs.sum(axis=1)
    return probs.div(row_sums, axis=0)

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
    regimes: pd.Series | None,
    probs_next: np.ndarray,
    forecast_mode: str,
    sample_weights=None,
    regime_probabilities: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Expected return vector using hard or soft regime information.

    In soft mode, if `regime_probabilities` is supplied, each regime mean is
    estimated with fuzzy historical memberships. The final forecast is then the
    expected value across possible next regimes:

        E[r_{t+1}] = sum_k P(regime_{t+1}=k) * E[r | regime=k]
    """
    probs = regime_weights(probs_next, forecast_mode)
    full_weights = _normalize_sample_weights(sample_weights, returns_df.index)
    regime_prob_frame = _normalize_regime_probability_frame(
        regime_probabilities,
        returns_df.index,
        n_regimes=len(probs),
    )

    weighted_sum = pd.Series(0.0, index=returns_df.columns, dtype=float)
    total_weight = 0.0

    for regime, next_weight in enumerate(probs):
        if next_weight <= 0:
            continue

        if regime_prob_frame is not None:
            membership = regime_prob_frame.iloc[:, regime]
            effective_weights = full_weights * membership
            if float(effective_weights.sum()) <= 0:
                continue
            regime_mean = _weighted_mean(returns_df, effective_weights)
        else:
            if regimes is None:
                continue
            subset = returns_df.loc[regimes == regime]
            if subset.empty:
                continue
            subset_weights = full_weights.loc[subset.index]
            regime_mean = _weighted_mean(subset, subset_weights)

        weighted_sum = weighted_sum.add(regime_mean * next_weight, fill_value=0.0)
        total_weight += float(next_weight)

    if total_weight <= 0:
        return _weighted_mean(returns_df, full_weights)
    return weighted_sum / total_weight


def _regime_weighted_std(
    returns_df: pd.DataFrame,
    regimes: pd.Series | None,
    probs_next: np.ndarray,
    forecast_mode: str,
    sample_weights=None,
    regime_probabilities: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    probs = regime_weights(probs_next, forecast_mode)
    full_weights = _normalize_sample_weights(sample_weights, returns_df.index)
    regime_prob_frame = _normalize_regime_probability_frame(
        regime_probabilities,
        returns_df.index,
        n_regimes=len(probs),
    )

    weighted_sum = pd.Series(0.0, index=returns_df.columns, dtype=float)
    total_weight = 0.0

    for regime, next_weight in enumerate(probs):
        if next_weight <= 0:
            continue

        if regime_prob_frame is not None:
            membership = regime_prob_frame.iloc[:, regime]
            effective_weights = full_weights * membership
            if float(effective_weights.sum()) <= 0:
                continue
            regime_std = _weighted_std(returns_df, effective_weights)
        else:
            if regimes is None:
                continue
            subset = returns_df.loc[regimes == regime]
            if subset.empty:
                continue
            subset_weights = full_weights.loc[subset.index]
            regime_std = _weighted_std(subset, subset_weights)

        weighted_sum = weighted_sum.add(regime_std * next_weight, fill_value=0.0)
        total_weight += float(next_weight)

    if total_weight <= 0:
        return _weighted_std(returns_df, full_weights)
    return weighted_sum / total_weight


def forecast_naive_sharpe(
    returns_df: pd.DataFrame,
    regimes: pd.Series | None,
    probs_next: np.ndarray,
    forecast_mode: str = "soft",
    sample_weights=None,
    regime_probabilities: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    mu = _regime_weighted_mean(
        returns_df,
        regimes,
        probs_next,
        forecast_mode,
        sample_weights=sample_weights,
        regime_probabilities=regime_probabilities,
    )
    sigma = _regime_weighted_std(
        returns_df,
        regimes,
        probs_next,
        forecast_mode,
        sample_weights=sample_weights,
        regime_probabilities=regime_probabilities,
    ).replace(0, 1e-8)
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




def _regime_view_forecast_error_cov_diag(
    returns_df: pd.DataFrame,
    regimes: pd.Series | None,
    forecast_mode: str,
    sample_weights=None,
    regime_probabilities: pd.DataFrame | np.ndarray | None = None,
    ridge: float = 1e-6,
) -> pd.DataFrame:
    """Diagonal Omega estimated from leave-one-out regime-view forecast errors.

    Black-Litterman view equation used here:

        q_t = P_t @ mu_regime,-t + epsilon_t

    In this project the views are asset-level absolute-return views, so the
    asset picking matrix is the identity. P_t is the row vector of historical
    regime memberships/probabilities for date t, and mu_regime,-t is the matrix
    of regime-conditional asset means estimated without using observation t.

    For each historical training observation t:

        q_t = sum_k P_t,k * E[r | regime=k, excluding t]
        epsilon_t = r_t - q_t

    Omega is diag(weighted Var(epsilon_t)). This avoids the in-sample leakage
    that would occur if r_t were used to estimate the same q_t it is evaluated
    against. It still uses only the rolling training window passed by the
    backtest; no future/current rebalance return is used here.
    """
    n_assets = returns_df.shape[1]
    if returns_df.empty:
        return pd.DataFrame(np.eye(n_assets) * ridge, index=returns_df.columns, columns=returns_df.columns)

    weights = _normalize_sample_weights(sample_weights, returns_df.index)
    returns_values = returns_df.to_numpy(dtype=float)
    weight_values = weights.to_numpy(dtype=float)

    if regime_probabilities is not None:
        n_regimes = np.asarray(regime_probabilities).shape[1]
    elif regimes is not None and len(regimes) > 0:
        n_regimes = int(pd.Series(regimes).dropna().max()) + 1
    else:
        # No regime information: use leave-one-out unconditional mean forecast.
        total_weight = float(weight_values.sum())
        total_sum = weight_values @ returns_values
        q_history_values = np.empty_like(returns_values)
        for i in range(len(returns_df)):
            denom = total_weight - weight_values[i]
            if denom > 1e-12:
                q_history_values[i] = (total_sum - weight_values[i] * returns_values[i]) / denom
            else:
                q_history_values[i] = total_sum / max(total_weight, 1e-12)
        errors = pd.DataFrame(returns_values - q_history_values, index=returns_df.index, columns=returns_df.columns)
        error_cov = _weighted_cov(errors, weights)
        error_variances = np.maximum(np.diag(error_cov.to_numpy(copy=True)), 0.0)
        omega = np.diag(error_variances + ridge)
        return pd.DataFrame(omega, index=returns_df.columns, columns=returns_df.columns)

    regime_prob_frame = _normalize_regime_probability_frame(
        regime_probabilities,
        returns_df.index,
        n_regimes=n_regimes,
    )

    if regime_prob_frame is not None and forecast_mode == "soft":
        # q_t = P_t @ mu_regime,-t, where P_t contains soft regime probabilities.
        membership_matrix = regime_prob_frame.to_numpy(dtype=float)
    else:
        # Hard view: P_t is one-hot for the assigned regime.
        if regimes is None:
            membership_matrix = np.ones((len(returns_df), n_regimes), dtype=float) / n_regimes
        else:
            hard_regimes = pd.Series(regimes).reindex(returns_df.index).fillna(0).astype(int).clip(0, n_regimes - 1)
            membership_matrix = np.zeros((len(returns_df), n_regimes), dtype=float)
            membership_matrix[np.arange(len(returns_df)), hard_regimes.to_numpy()] = 1.0

    # Precompute weighted regime sums over the training window. For each date t
    # below, subtract t's own weighted contribution before forming q_t. This is
    # the leave-one-out step that prevents in-sample target leakage in Omega.
    effective_weights = membership_matrix * weight_values[:, None]
    regime_denoms = effective_weights.sum(axis=0)  # shape: (n_regimes,)
    regime_sums = effective_weights.T @ returns_values  # shape: (n_regimes, n_assets)

    total_weight = float(weight_values.sum())
    total_sum = weight_values @ returns_values

    q_history_values = np.empty_like(returns_values)
    for i in range(len(returns_df)):
        row_membership = membership_matrix[i]
        loo_regime_means = np.empty((n_regimes, n_assets), dtype=float)

        # Fallback is also leave-one-out, so the observed r_t is not used in its
        # own forecast even when a regime has no other effective observations.
        fallback_denom = total_weight - weight_values[i]
        if fallback_denom > 1e-12:
            fallback_mean = (total_sum - weight_values[i] * returns_values[i]) / fallback_denom
        else:
            fallback_mean = total_sum / max(total_weight, 1e-12)

        for regime in range(n_regimes):
            own_effective_weight = weight_values[i] * membership_matrix[i, regime]
            denom = regime_denoms[regime] - own_effective_weight
            if denom > 1e-12:
                numerator = regime_sums[regime] - own_effective_weight * returns_values[i]
                loo_regime_means[regime] = numerator / denom
            else:
                loo_regime_means[regime] = fallback_mean

        q_history_values[i] = row_membership @ loo_regime_means

    errors = pd.DataFrame(
        returns_values - q_history_values,
        index=returns_df.index,
        columns=returns_df.columns,
    )

    # Omega = diag(Cov(epsilon_t)). Use the same time-decay/sample weights as
    # the rest of the forecasting code, so recent forecast errors can matter more.
    error_cov = _weighted_cov(errors, weights)
    error_variances = np.maximum(np.diag(error_cov.to_numpy(copy=True)), 0.0)
    omega = np.diag(error_variances + ridge)
    return pd.DataFrame(omega, index=returns_df.columns, columns=returns_df.columns)

def forecast_black_litterman_scores(
    returns_df: pd.DataFrame,
    regimes: pd.Series | None,
    probs_next: np.ndarray,
    tau: float = BLACK_LITTERMAN_TAU,
    forecast_mode: str = "soft",
    sample_weights=None,
    regime_probabilities: pd.DataFrame | np.ndarray | None = None,
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
        regime_probabilities=regime_probabilities,
    ).to_numpy()

    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma)

    # View uncertainty Omega is estimated from historical regime-view forecast
    # errors instead of being tied mechanically to tau * Sigma.
    #
    # View equation: q_t = P_t @ mu_regime + epsilon_t.
    # Since current views are asset-level absolute-return views, P_asset = I.
    # We therefore estimate Omega as diag(weighted Cov(epsilon_t)).
    omega = _regime_view_forecast_error_cov_diag(
        returns_df,
        regimes,
        forecast_mode,
        sample_weights=sample_weights,
        regime_probabilities=regime_probabilities,
    ).to_numpy(copy=True)
    omega_inv = np.linalg.pinv(omega)

    posterior = np.linalg.pinv(tau_sigma_inv + omega_inv) @ (tau_sigma_inv @ mu_prior + omega_inv @ q_star)
    return pd.Series(posterior, index=returns_df.columns)


def forecast_mvo_scores(
    returns_df: pd.DataFrame,
    regimes: pd.Series | None = None,
    probs_next: np.ndarray | None = None,
    forecast_mode: str = "soft",
    sample_weights=None,
    regime_probabilities: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Mean-variance scores using expected regime-weighted returns.

    If regimes and next-regime probabilities are supplied, the expected return
    vector is blended across regimes. In soft mode this is a probability-weighted
    expected value; in hard mode it uses only the most likely next regime.
    The covariance matrix is still estimated from the full rolling window for
    stability.
    """
    if regimes is not None and probs_next is not None:
        mu_series = _regime_weighted_mean(
            returns_df,
            regimes,
            probs_next,
            forecast_mode,
            sample_weights=sample_weights,
            regime_probabilities=regime_probabilities,
        )
    else:
        mu_series = _weighted_mean(returns_df, sample_weights)

    mu = mu_series.to_numpy()
    sigma = _weighted_cov(returns_df, sample_weights).to_numpy(copy=True) + np.eye(returns_df.shape[1]) * 1e-6
    scores = np.linalg.pinv(sigma) @ mu
    return pd.Series(scores, index=returns_df.columns)


def compute_random_regime_state(index: pd.Index, n_regimes: int, rng: np.random.Generator) -> tuple[pd.Series, pd.DataFrame]:
    random_regimes = pd.Series(rng.integers(0, n_regimes, size=len(index)), index=index, name="regime")
    probs = np.zeros((len(index), n_regimes), dtype=float)
    probs[np.arange(len(index)), random_regimes.to_numpy()] = 1.0
    probs_df = pd.DataFrame(probs, index=index, columns=[f"regime_prob_{i}" for i in range(n_regimes)])
    return random_regimes, probs_df
