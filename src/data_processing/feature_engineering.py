from __future__ import annotations

import pandas as pd


def apply_feature_engineering(
    df: pd.DataFrame,
    ema_span: int = 12,
) -> pd.DataFrame:
    """
    Augment df with EMA and first-order difference columns.
    Returns the original columns plus _ema and _diff suffixed columns.
    The first row is dropped because diff() produces NaN there.
    """
    ema = df.ewm(span=ema_span, adjust=False).mean().add_suffix("_ema")
    diff = df.diff(1).add_suffix("_diff")
    out = pd.concat([df, ema, diff], axis=1)
    return out.iloc[1:].copy()
