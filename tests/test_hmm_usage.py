"""Usage example for GaussianHMMRegimeModel."""

import numpy as np
import pandas as pd

from macro_prediction.hmm import GaussianHMMRegimeModel


def test_gaussian_hmm_regime_model_usage_example() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-31", periods=36, freq="ME")
    features = pd.DataFrame(
        np.vstack([
            rng.normal(-2.0, 0.2, size=(12, 2)),
            rng.normal(0.0, 0.2, size=(12, 2)),
            rng.normal(2.0, 0.2, size=(12, 2)),
        ]),
        index=index,
        columns=["growth", "inflation"],
    )

    model = GaussianHMMRegimeModel(n_regimes=3, random_state=42, n_iter=200)
    result = model.fit_predict(features)
    next_month = model.predict_next(features)

    assert result.regimes.index.equals(features.index)
    assert result.probabilities.shape == (36, 3)
    assert next_month.index.tolist() == ["prob_regime_0", "prob_regime_1", "prob_regime_2"]
    assert abs(next_month.sum() - 1.0) < 1e-12
