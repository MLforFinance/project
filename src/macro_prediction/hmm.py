"""Gaussian HMM regime model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


@dataclass(frozen=True)
class RegimeResult:
    """Regime labels and probabilities indexed like the input features."""

    regimes: pd.Series
    probabilities: pd.DataFrame


class GaussianHMMRegimeModel:
    """Small wrapper around hmmlearn's GaussianHMM for macro regimes."""

    def __init__(
        self,
        n_regimes: int = 3,
        *,
        covariance_type: str = "full",
        n_iter: int = 500,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self.columns_: list[str] | None = None

    def fit(self, features: pd.DataFrame) -> "GaussianHMMRegimeModel":
        """Fit the HMM on transformed/scaled feature rows."""

        clean = self._clean_features(features)
        self.columns_ = list(clean.columns)
        self.model.fit(clean.to_numpy())
        self._repair_transition_matrix()
        return self

    def predict(self, features: pd.DataFrame) -> RegimeResult:
        """Assign each row to its most likely hidden regime."""

        clean = self._prepare_features(features)
        states = self.model.predict(clean.to_numpy())
        probabilities = self.model.predict_proba(clean.to_numpy())
        return RegimeResult(
            regimes=pd.Series(states, index=clean.index, name="regime"),
            probabilities=self._probability_frame(probabilities, clean.index),
        )

    def predict_next(self, features: pd.DataFrame) -> pd.Series:
        """Estimate next-row regime probabilities from the latest observed row."""

        current = self.predict(features).probabilities.iloc[-1].to_numpy()
        next_probabilities = current @ self.model.transmat_
        return pd.Series(
            next_probabilities,
            index=[f"prob_regime_{i}" for i in range(self.n_regimes)],
            name="next_regime_probability",
        )

    def fit_predict(self, features: pd.DataFrame) -> RegimeResult:
        """Fit the model, then assign regimes to the same feature matrix."""

        self.fit(features)
        return self.predict(features)

    def _prepare_features(self, features: pd.DataFrame) -> pd.DataFrame:
        clean = self._clean_features(features)
        if self.columns_ is not None:
            clean = clean.reindex(columns=self.columns_)
        return clean

    @staticmethod
    def _clean_features(features: pd.DataFrame) -> pd.DataFrame:
        clean = features.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
        if clean.empty:
            raise ValueError("Cannot fit or predict regimes with no complete feature rows.")
        return clean.astype(float)

    def _probability_frame(self, probabilities: np.ndarray, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(
            probabilities,
            index=index,
            columns=[f"prob_regime_{i}" for i in range(self.n_regimes)],
        )

    def _repair_transition_matrix(self) -> None:
        """Keep hmmlearn usable if a fitted state has no observed outgoing moves."""

        row_sums = self.model.transmat_.sum(axis=1)
        bad_rows = np.where(row_sums == 0)[0]
        if len(bad_rows) == 0:
            return
        for row in bad_rows:
            self.model.transmat_[row] = np.full(self.n_regimes, 1.0 / self.n_regimes)
