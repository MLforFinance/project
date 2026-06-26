# Macroeconomic Prediction

Clean starting point for macroeconomic prediction experiments.

## FRED feature builder

`FredFeatureBuilder` fetches live FRED series with `fredapi`, converts each series to monthly observations, applies per-series transformations, and standard-scales the feature matrix.

```python
from macro_prediction.fred import FredFeatureBuilder, default_fredmd_representative_series

series = default_fredmd_representative_series()
builder = FredFeatureBuilder(series)  # reads FRED_API_KEY from the environment

raw = builder.fetch_raw(
    observation_start="1985-01-01",
    vintage_date="2024-12-31",
)
features = builder.fit_transform(raw)
```

Set your API key before running Python:

```bash
export FRED_API_KEY="your_key_here"
```

For forecasting, fit only on the training window, then transform later months with the fitted scaler.

```python
train_features = builder.fit_transform(train_raw)
test_features = builder.transform(test_raw)
```

Passing `vintage_date` sends it to FRED as both `realtime_start` and `realtime_end`, so later pipeline steps can use data as available at that vintage date.

The default representative set contains: `RPI`, `INDPRO`, `CUMFNS`, `PAYEMS`, `UNRATE`, `ICSA`, `HOUST`, `PERMIT`, `RSAFS`, `DGORDER`, `CPIAUCSL`, `PCEPI`, `PPIACO`, `M2SL`, `BUSLOANS`, `FEDFUNDS`, `GS10`, and `T10Y3M`.

## Gaussian HMM regimes

After building transformed and standardized FRED features, fit a 3-regime Gaussian HMM with `hmmlearn`:

```python
from macro_prediction.hmm import GaussianHMMRegimeModel

hmm = GaussianHMMRegimeModel(n_regimes=3, random_state=42)
result = hmm.fit_predict(features)

regime_by_month = result.regimes
regime_probabilities = result.probabilities
next_month_probabilities = hmm.predict_next(features)
```

`result.regimes` is the max-probability regime for each month. `result.probabilities` keeps the full regime probability vector for each month.

## Main training run

Set your API key, then run the simple main module:

```bash
export FRED_API_KEY="your_key_here"
python -m macro_prediction.main
```

The main run trains a 4-state Gaussian HMM in two ways:

- `expanding`: each prediction uses all prior months available up to that point.
- `rolling_10y`: each prediction uses only the prior 120 months.

It writes CSV outputs under `artifacts/hmm_macro_regimes/`:

- `raw_macro_data.csv`
- `expanding_walk_forward_predictions.csv`
- `expanding_latest_next_month_probabilities.csv`
- `rolling_10y_walk_forward_predictions.csv`
- `rolling_10y_latest_next_month_probabilities.csv`
- `expanding_regime_transitions.png`
- `rolling_10y_regime_transitions.png`

## Setup

```bash
python -m pip install -e .
```
