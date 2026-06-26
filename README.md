# Macroeconomic Prediction

Clean starting point for macroeconomic prediction experiments.

## FRED feature builder

`FredFeatureBuilder` fetches live FRED series with `fredapi`, converts each series to monthly observations, applies per-series transformations, and standard-scales the feature matrix.

```python
from macro_prediction.fred import FredFeatureBuilder, FredSeriesSpec

series = [
    # Monthly level series: keep the monthly observation.
    FredSeriesSpec("INDPRO", transformation="log_diff", frequency="monthly"),
    FredSeriesSpec("PAYEMS", transformation="log_diff", frequency="monthly"),
    FredSeriesSpec("UNRATE", transformation="diff", frequency="monthly"),
    FredSeriesSpec("CPIAUCSL", transformation="yoy_log_diff", frequency="monthly"),

    # Daily/weekly series: aggregate actual observations inside each month.
    FredSeriesSpec(
        "T10Y3M",
        transformation="level",
        frequency="daily_or_weekly",
        monthly_aggregation="mean",
    ),

    # Quarterly series: repeat the quarter value into each month of the quarter.
    FredSeriesSpec("GDPC1", transformation="log_diff", frequency="quarterly"),
]

builder = FredFeatureBuilder(series, api_key="YOUR_FRED_API_KEY")
raw = builder.fetch_raw(
    observation_start="1985-01-01",
    vintage_date="2024-12-31",
)
features = builder.fit_transform(raw)
```

For forecasting, fit only on the training window, then transform later months with the fitted scaler.

```python
train_features = builder.fit_transform(train_raw)
test_features = builder.transform(test_raw)
```

Passing `vintage_date` sends it to FRED as both `realtime_start` and `realtime_end`, so later pipeline steps can use data as available at that vintage date.

## Setup

```bash
python -m pip install -e .
```
