from __future__ import annotations

import pandas as pd


# def standardize_scores(scores: pd.Series, fallback_index: pd.Index) -> pd.Series:
#     standardized = pd.Series(scores, index=fallback_index, dtype=float)
#     return standardized.fillna(0.0)
def standardize_scores(scores: pd.Series, fallback_index: pd.Index) -> pd.Series:
    return scores.reindex(fallback_index).fillna(0.0)


def position_weights(predictions: pd.Series, mode: str, l_value: int, predicted_regime: int | None = None) -> pd.Series:
    preds = predictions.astype(float)
    if mode == "mx":
        actual_mode = "los" if predicted_regime == 0 else "lo"
        return position_weights(preds, actual_mode, l_value, predicted_regime=predicted_regime)

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
        selected = preds.loc[preds.abs().nlargest(
            min(l_value, len(preds))).index]
        denom = selected.abs().sum()
        if denom > 0:
            weights.loc[selected.index] = selected / denom
        return weights

    raise ValueError("mode must be one of: 'lo', 'lns', 'los', 'mx'.")
