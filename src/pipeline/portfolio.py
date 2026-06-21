from __future__ import annotations

import pandas as pd


def standardize_scores(scores: pd.Series, fallback_index: pd.Index) -> pd.Series:
    return scores.reindex(fallback_index).fillna(0.0)


def evolve_weights(weights: pd.Series, asset_returns: pd.Series) -> pd.Series:
    standardized_weights = standardize_scores(weights.astype(float), asset_returns.index)
    standardized_returns = asset_returns.reindex(standardized_weights.index).fillna(0.0).astype(float)
    portfolio_return = float(standardized_weights.dot(standardized_returns))
    denominator = 1.0 + portfolio_return
    if abs(denominator) < 1e-12:
        return standardized_weights
    evolved = standardized_weights * (1.0 + standardized_returns) / denominator
    return evolved.reindex(standardized_weights.index).fillna(0.0)


def traded_notional(current_weights: pd.Series, target_weights: pd.Series) -> float:
    standardized_target = target_weights.astype(float)
    standardized_current = standardize_scores(current_weights.astype(float), standardized_target.index)
    return float((standardized_target - standardized_current).abs().sum())


def transaction_cost(current_weights: pd.Series, target_weights: pd.Series, cost_rate: float) -> float:
    return traded_notional(current_weights, target_weights) * float(cost_rate)


def position_weights(
    predictions: pd.Series,
    mode: str,
    l_value: int,
    predicted_regime: int | None = None,
    regime_probs: pd.Series | None = None,
) -> pd.Series:
    preds = predictions.astype(float)
    if mode == "mx":
        if regime_probs is None:
            actual_mode = "los" if predicted_regime == 0 else "lo"
            return position_weights(preds, actual_mode, l_value, predicted_regime=predicted_regime)

        recession_weight = float(regime_probs.get(0, 0.0))
        los_weights = position_weights(preds, "los", l_value, predicted_regime=predicted_regime)
        lo_weights = position_weights(preds, "lo", l_value, predicted_regime=predicted_regime)
        return recession_weight * los_weights + (1.0 - recession_weight) * lo_weights

    weights = pd.Series(0.0, index=preds.index)

    if mode == "lo":
        top = preds.nlargest(min(l_value, len(preds)))
        selected = top.clip(lower=0.0)
        denom = selected.sum()
        if denom <= 0:
            weights.loc[top.index] = 1.0 / len(top)
        else:
            weights.loc[top.index] = selected / denom
        return weights

    if mode == "lns":
        top = preds.nlargest(min(l_value, len(preds)))
        bottom = preds.nsmallest(min(l_value, len(preds)))
        selected = pd.concat([top, bottom[~bottom.index.isin(top.index)]])
        denom = selected.abs().sum()
        if denom > 0:
            weights.loc[selected.index] = selected / denom
        return weights

    if mode == "los":
        selected = preds.loc[preds.abs().nlargest(min(l_value, len(preds))).index]
        denom = selected.abs().sum()
        if denom > 0:
            weights.loc[selected.index] = selected / denom
        return weights

    raise ValueError("mode must be one of: 'lo', 'lns', 'los', 'mx'.")
