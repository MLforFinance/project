# Macroeconomic Prediction

Focused project for one model: a walk-forward logistic regression that estimates the probability that next month's S&P 500 return will be negative, then turns that probability into an S&P 500 / T-bill allocation.

## Setup

```bash
python -m pip install -e .
export FRED_API_KEY="your_key_here"
python -m macro_prediction.main
```

## Model

The model uses live FRED data through `fredapi`, plus lagged S&P 500 market features from Yahoo Finance. The default macro basket is intentionally small:

`INDPRO`, `PAYEMS`, `UNRATE`, `ICSA`, `HOUST`, `RSAFS`, `CPIAUCSL`, `M2SL`, `FEDFUNDS`, and `T10Y3M`.

For each prediction month, the classifier only trains on prior months. The target is whether the following month's S&P 500 return is below `target_return_threshold`, which defaults to `0.0`.

All configurable model and strategy parameters live in `ModelConfig` inside `src/macro_prediction/main.py`.

Important defaults:

- `mode="expanding"`: train on all prior months.
- `min_train_months=120`: wait for 10 years of training data before predicting.
- `include_market_features=True`: add lagged S&P 500 return, volatility, and drawdown features.
- `allocation_rule="bands"`: map predicted risk into position sizes.
- `lower_probability_threshold=0.33`: below this, hold 100% S&P 500.
- `upper_probability_threshold=0.67`: above this, hold 100% T-bills.
- `mid_equity_weight=0.50`: between the thresholds, hold 50% S&P 500 and 50% T-bills.
- `cost_per_unit_turnover=0.001`: charge 10 bps per full unit of equity turnover.

## Outputs

The run writes outputs under `artifacts/negative_month_model/`:

- `raw_macro_data.csv`
- `sp500_monthly_returns.csv`
- `tbill_cash_returns.csv`
- `expanding_bad_month_predictions.csv`
- `expanding_bad_month_summary.csv`
- `expanding_bad_month_probabilities.png`
- `strategy_returns.csv`
- `strategy_summary.csv`
- `strategy_equity_drawdown.png`
