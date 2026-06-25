from __future__ import annotations

import pandas as pd


CASH_TICKER = "CASH"


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


def traded_notional(
    current_weights: pd.Series,
    target_weights: pd.Series,
    cash_ticker: str | None = CASH_TICKER,
) -> float:
    """Return risky-asset turnover.

    The cash leg is excluded because moving money into or out of the cash
    balance is the residual of trading risky assets, not a separately traded ETF.
    """
    standardized_target = target_weights.astype(float)
    standardized_current = standardize_scores(current_weights.astype(float), standardized_target.index)
    diff = standardized_target - standardized_current
    if cash_ticker is not None and cash_ticker in diff.index:
        diff = diff.drop(cash_ticker)
    return float(diff.abs().sum())


def transaction_cost(
    current_weights: pd.Series,
    target_weights: pd.Series,
    cost_rate: float,
    cash_ticker: str | None = CASH_TICKER,
) -> float:
    return traded_notional(current_weights, target_weights, cash_ticker=cash_ticker) * float(cost_rate)


def add_cash_asset(scores: pd.Series, cash_ticker: str = CASH_TICKER, cash_score: float = 0.0) -> pd.Series:
    """Add a zero-return cash alternative to a score vector."""
    scores = scores.astype(float).copy()
    if cash_ticker not in scores.index:
        scores.loc[cash_ticker] = float(cash_score)
    return scores


def apply_fixed_risk_overlay(
    weights: pd.Series,
    exposure: float,
    cash_ticker: str = CASH_TICKER,
) -> pd.Series:
    """Scale risky weights and hold the unused capital in 0%-return cash.

    exposure=1.00 means no overlay. exposure=0.70 means 70% invested in
    risky assets and 30% held as cash. Cash from the optimizer itself is
    preserved and then receives the residual balance.
    """
    exposure = max(0.0, min(float(exposure), 1.0))
    weights = weights.astype(float).copy()
    if cash_ticker not in weights.index:
        weights.loc[cash_ticker] = 0.0

    cash_before = float(weights.get(cash_ticker, 0.0))
    risky = weights.drop(cash_ticker)
    scaled = risky * exposure

    out = pd.Series(0.0, index=weights.index, dtype=float)
    out.loc[scaled.index] = scaled
    out.loc[cash_ticker] = 1.0 - float(scaled.sum())

    # If the original optimizer already chose cash, this keeps cash as the
    # residual balance rather than forcing a separate cash trade.
    if cash_before > 0 and exposure >= 1.0:
        out.loc[cash_ticker] = cash_before

    return out


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
            # If the synthetic cash asset is one of the best available assets,
            # avoid forcing capital into ETFs with non-positive scores.
            if CASH_TICKER in top.index:
                weights.loc[CASH_TICKER] = 1.0
            else:
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
