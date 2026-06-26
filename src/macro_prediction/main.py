"""Train 4-state HMM macro regime predictions.

Run with:

    python -m macro_prediction.main

Configuration is intentionally simple and kept near the top of this file. Set
FRED_API_KEY in the environment before running.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from macro_prediction.fred import (
    FredFeatureBuilder,
    FredSeriesSpec,
    default_fredmd_representative_series,
)
from macro_prediction.hmm import GaussianHMMRegimeModel
from macro_prediction.plotting import plot_regime_transitions

OBSERVATION_START = "1985-01-01"
OBSERVATION_END = None
VINTAGE_DATE = None
N_REGIMES = 4
MIN_TRAIN_MONTHS = 120
ROLLING_WINDOW_MONTHS = 120
OUTPUT_DIR = Path("artifacts/hmm_macro_regimes")


@dataclass(frozen=True)
class PredictionRun:
    """Outputs from one HMM prediction mode."""

    mode: str
    predictions: pd.DataFrame
    latest_next_month: pd.Series


def main() -> None:
    """Fetch macro data and train 4-state HMM regime predictors."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    series = default_fredmd_representative_series()
    builder = FredFeatureBuilder(series)

    raw = builder.fetch_raw(
        observation_start=OBSERVATION_START,
        observation_end=OBSERVATION_END,
        vintage_date=VINTAGE_DATE,
    )
    raw.to_csv(OUTPUT_DIR / "raw_macro_data.csv")

    expanding = walk_forward_next_month_predictions(
        raw,
        series,
        mode="expanding",
        n_regimes=N_REGIMES,
        min_train_months=MIN_TRAIN_MONTHS,
    )
    rolling_10y = walk_forward_next_month_predictions(
        raw,
        series,
        mode="rolling_10y",
        n_regimes=N_REGIMES,
        min_train_months=MIN_TRAIN_MONTHS,
        window_months=ROLLING_WINDOW_MONTHS,
    )

    save_prediction_run(expanding)
    save_prediction_run(rolling_10y)

    print("Saved outputs to", OUTPUT_DIR)
    print("Latest expanding next-month probabilities:")
    print(expanding.latest_next_month.to_string())
    print("Latest rolling-10y next-month probabilities:")
    print(rolling_10y.latest_next_month.to_string())


def walk_forward_next_month_predictions(
    raw: pd.DataFrame,
    series: list[FredSeriesSpec],
    *,
    mode: str,
    n_regimes: int,
    min_train_months: int,
    window_months: int | None = None,
) -> PredictionRun:
    """Predict each next month using only raw data available through prior months."""

    if mode not in {"expanding", "rolling_10y"}:
        raise ValueError("mode must be 'expanding' or 'rolling_10y'")
    if mode == "rolling_10y" and window_months is None:
        raise ValueError("rolling_10y mode requires window_months")

    if len(raw) <= min_train_months:
        raise ValueError(
            f"Need more than {min_train_months} monthly rows for walk-forward prediction; "
            f"got {len(raw)}."
        )

    rows = []
    for train_end_position in range(min_train_months, len(raw)):
        train_raw = raw.iloc[:train_end_position]
        if mode == "rolling_10y":
            train_raw = train_raw.iloc[-window_months:]

        prediction = fit_and_predict_next_month(
            train_raw,
            series,
            n_regimes=n_regimes,
        )
        prediction_month = raw.index[train_end_position]
        rows.append(
            {
                "prediction_month": prediction_month,
                "trained_through": raw.index[train_end_position - 1],
                "training_months": len(train_raw),
                **prediction.to_dict(),
                "predicted_regime": int(prediction.idxmax().replace("prob_regime_", "")),
            }
        )

    predictions = pd.DataFrame(rows).set_index("prediction_month")

    latest_raw = raw if mode == "expanding" else raw.iloc[-window_months:]
    latest_next_month = fit_and_predict_next_month(
        latest_raw,
        series,
        n_regimes=n_regimes,
    )
    latest_next_month.name = next_month_end(raw.index[-1])

    return PredictionRun(
        mode=mode,
        predictions=predictions,
        latest_next_month=latest_next_month,
    )


def fit_and_predict_next_month(
    train_raw: pd.DataFrame,
    series: list[FredSeriesSpec],
    *,
    n_regimes: int,
) -> pd.Series:
    """Fit scaler plus HMM on one training window and predict the next regime."""

    builder = FredFeatureBuilder(series)
    features = builder.fit_transform(train_raw)

    model = GaussianHMMRegimeModel(
        n_regimes=n_regimes,
        covariance_type="diag",
        random_state=42,
    )
    model.fit(features)
    return model.predict_next(features)


def save_prediction_run(run: PredictionRun) -> None:
    """Save one prediction mode to CSV files."""

    run.predictions.to_csv(OUTPUT_DIR / f"{run.mode}_walk_forward_predictions.csv")
    run.latest_next_month.to_frame("probability").to_csv(
        OUTPUT_DIR / f"{run.mode}_latest_next_month_probabilities.csv"
    )
    plot_regime_transitions(
        run.predictions,
        OUTPUT_DIR / f"{run.mode}_regime_transitions.png",
        title=f"Predicted macro regime over time ({run.mode})",
    )


def next_month_end(date: pd.Timestamp) -> pd.Timestamp:
    """Return the month-end timestamp after the given month."""

    return (date.to_period("M") + 1).to_timestamp("M")


if __name__ == "__main__":
    main()
