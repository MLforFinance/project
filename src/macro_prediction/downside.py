"""Walk-forward model for predicting negative S&P 500 months."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from macro_prediction.fred import FredFeatureBuilder, FredSeriesSpec


@dataclass(frozen=True)
class BadMonthRun:
    """Outputs from one bad-month prediction mode."""

    mode: str
    predictions: pd.DataFrame
    summary: pd.Series


def fetch_sp500_monthly_returns(start: str = "1960-01-01") -> pd.Series:
    """Fetch monthly S&P 500 returns from Yahoo Finance."""

    prices = yf.download("^GSPC", start=start, progress=False, auto_adjust=False)
    if prices.empty:
        raise ValueError("No S&P 500 data returned from Yahoo Finance.")
    close = prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    monthly_close = close.resample("ME").last().dropna()
    returns = monthly_close.pct_change().dropna()
    returns.name = "sp500_return"
    return returns


def next_month_end(date: pd.Timestamp) -> pd.Timestamp:
    """Return the month-end timestamp after the given month."""

    return (date.to_period("M") + 1).to_timestamp("M")


def walk_forward_bad_month_predictions(
    raw: pd.DataFrame,
    series: list[FredSeriesSpec],
    sp500_returns: pd.Series,
    *,
    mode: str,
    bad_return_threshold: float = -0.05,
    min_train_months: int = 120,
    window_months: int | None = None,
    probability_threshold: float = 0.50,
    include_market_features: bool = False,
) -> BadMonthRun:
    """Predict whether each next month will be a negative S&P 500 month.

    For prediction month T, the model only sees macro data through T-1. Training
    examples use macro features from month M to predict S&P 500 return in M+1.
    """

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
        prediction_month = raw.index[train_end_position]
        trained_through = raw.index[train_end_position - 1]
        train_raw = raw.iloc[:train_end_position]
        if mode == "rolling_10y":
            train_raw = train_raw.iloc[-window_months:]

        probability = fit_and_predict_bad_month_probability(
            train_raw,
            series,
            sp500_returns,
            trained_through=trained_through,
            bad_return_threshold=bad_return_threshold,
            include_market_features=include_market_features,
        )
        actual_return = sp500_returns.get(prediction_month, pd.NA)
        actual_bad_month = (actual_return <= bad_return_threshold) if pd.notna(actual_return) else pd.NA
        rows.append(
            {
                "prediction_month": prediction_month,
                "trained_through": trained_through,
                "training_months": len(train_raw),
                "bad_month_probability": probability,
                "predicted_bad_month": probability >= probability_threshold,
                "actual_sp500_return": actual_return,
                "actual_bad_month": actual_bad_month,
            }
        )

    predictions = pd.DataFrame(rows).set_index("prediction_month")
    summary = summarize_bad_month_predictions(predictions, probability_threshold=probability_threshold)
    return BadMonthRun(mode=mode, predictions=predictions, summary=summary)


def fit_and_predict_bad_month_probability(
    train_raw: pd.DataFrame,
    series: list[FredSeriesSpec],
    sp500_returns: pd.Series,
    *,
    trained_through: pd.Timestamp,
    bad_return_threshold: float,
    include_market_features: bool = False,
) -> float:
    """Fit one walk-forward classifier and predict the next month's bad-month risk."""

    builder = FredFeatureBuilder(series)
    features = builder.fit_transform(train_raw)
    if include_market_features:
        features = add_market_features(features, sp500_returns)
    features = standardize_feature_frame(features)
    x_train, y_train = make_supervised_bad_month_dataset(
        features,
        sp500_returns,
        trained_through=trained_through,
        bad_return_threshold=bad_return_threshold,
    )
    if len(y_train) == 0:
        return 0.0
    if y_train.nunique() == 1:
        return float(y_train.iloc[0])

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    latest_features = features.iloc[[-1]]
    return float(model.predict_proba(latest_features)[0, 1])


def add_market_features(features: pd.DataFrame, sp500_returns: pd.Series) -> pd.DataFrame:
    """Add market features known as of each feature month.

    Each row at month M uses S&P 500 data through month M only. That row is then
    used to predict the S&P 500 return in month M+1.
    """

    returns = sp500_returns.reindex(features.index)
    market = pd.DataFrame(index=features.index)
    market["sp500_return_1m"] = returns
    market["sp500_return_3m"] = (1.0 + returns).rolling(3).apply(lambda x: x.prod() - 1.0, raw=True)
    market["sp500_return_6m"] = (1.0 + returns).rolling(6).apply(lambda x: x.prod() - 1.0, raw=True)
    market["sp500_return_12m"] = (1.0 + returns).rolling(12).apply(lambda x: x.prod() - 1.0, raw=True)
    market["sp500_vol_3m"] = returns.rolling(3).std()
    market["sp500_vol_6m"] = returns.rolling(6).std()
    market["sp500_vol_12m"] = returns.rolling(12).std()

    monthly_index = (1.0 + returns).cumprod()
    trailing_peak = monthly_index.cummax()
    market["sp500_drawdown"] = monthly_index / trailing_peak - 1.0

    combined = features.join(market)
    return combined.dropna(axis=0, how="any")


def standardize_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Standard-scale all model features within the current training window."""

    scaler = StandardScaler()
    values = scaler.fit_transform(features)
    return pd.DataFrame(values, index=features.index, columns=features.columns)


def make_supervised_bad_month_dataset(
    features: pd.DataFrame,
    sp500_returns: pd.Series,
    *,
    trained_through: pd.Timestamp,
    bad_return_threshold: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align macro feature month M with S&P 500 return in month M+1."""

    rows = []
    targets = []
    for feature_month, row in features.iterrows():
        target_month = next_month_end(feature_month)
        if target_month > trained_through:
            continue
        if target_month not in sp500_returns.index:
            continue
        target_return = sp500_returns.loc[target_month]
        if pd.isna(target_return):
            continue
        rows.append(row)
        targets.append(int(target_return <= bad_return_threshold))

    if not rows:
        return pd.DataFrame(columns=features.columns), pd.Series(dtype=int, name="bad_month")
    return pd.DataFrame(rows), pd.Series(targets, index=[row.name for row in rows], name="bad_month")


def summarize_bad_month_predictions(
    predictions: pd.DataFrame,
    *,
    probability_threshold: float,
) -> pd.Series:
    """Summarize bad-month classifier performance."""

    scored = predictions.dropna(subset=["actual_bad_month", "actual_sp500_return"]).copy()
    if scored.empty:
        return pd.Series(dtype=float)

    predicted = scored["bad_month_probability"] >= probability_threshold
    actual = scored["actual_bad_month"].astype(bool)
    true_positive = int((predicted & actual).sum())
    false_positive = int((predicted & ~actual).sum())
    false_negative = int((~predicted & actual).sum())
    true_negative = int((~predicted & ~actual).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    flagged = scored.loc[predicted, "actual_sp500_return"]
    unflagged = scored.loc[~predicted, "actual_sp500_return"]

    return pd.Series(
        {
            "months": len(scored),
            "actual_bad_months": int(actual.sum()),
            "predicted_bad_months": int(predicted.sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": precision,
            "recall": recall,
            "avg_return_flagged_pct": flagged.mean() * 100 if len(flagged) else pd.NA,
            "avg_return_unflagged_pct": unflagged.mean() * 100 if len(unflagged) else pd.NA,
            "vol_flagged_pct": flagged.std() * 100 if len(flagged) > 1 else pd.NA,
            "vol_unflagged_pct": unflagged.std() * 100 if len(unflagged) > 1 else pd.NA,
        }
    )


def save_bad_month_run(run: BadMonthRun, output_dir: str | Path) -> None:
    """Save predictions, summary, and a probability plot."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run.predictions.to_csv(output / f"{run.mode}_bad_month_predictions.csv")
    run.summary.to_frame("value").to_csv(output / f"{run.mode}_bad_month_summary.csv")
    plot_bad_month_probabilities(
        run.predictions,
        output / f"{run.mode}_bad_month_probabilities.png",
        title=f"Predicted probability of negative S&P 500 month ({run.mode})",
    )


def plot_bad_month_probabilities(
    predictions: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot bad-month probability through time with actual bad months marked."""

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        predictions.index,
        predictions["bad_month_probability"],
        color="#1f77b4",
        linewidth=1.7,
        label="Predicted bad-month probability",
    )
    actual_bad = predictions[predictions["actual_bad_month"] == True]
    if not actual_bad.empty:
        ax.scatter(
            actual_bad.index,
            actual_bad["bad_month_probability"],
            color="#d62728",
            s=18,
            label="Actual bad month",
            zorder=3,
        )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
