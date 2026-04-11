from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import build_metrics_table
from .config import DEFAULT_L_VALUES, DEFAULT_SIZING_MODES, MODEL_FAMILIES, RANDOM_SEED
from .forecasting import (
    compute_random_regime_state,
    forecast_black_litterman_scores,
    forecast_mvo_scores,
    forecast_naive_sharpe,
    predict_ridge,
    train_ridge_models,
)
from .portfolio import position_weights, standardize_scores
from .regime_pipeline import compute_transition_matrix, compute_window_regime_state, next_regime_probs, renormalize_probabilities
from .reporting import flatten_panel


def run_walk_forward_backtest(
    X_full: pd.DataFrame,
    Y_targets: pd.DataFrame,
    target_dates: pd.Index,
    regime_count: int,
    window_size: int,
    ridge_alpha: float,
    l_values: tuple[int, ...] = DEFAULT_L_VALUES,
    sizing_modes: tuple[str, ...] = DEFAULT_SIZING_MODES,
) -> dict[str, object]:
    if len(X_full) < window_size:
        raise ValueError(f"Not enough aligned observations for a {window_size}-month window. Found {len(X_full)} rows.")

    n_regimes = regime_count + 1
    asset_columns = list(Y_targets.columns)
    benchmark_col = "SPY" if "SPY" in asset_columns else asset_columns[0]
    rng = np.random.default_rng(RANDOM_SEED)

    strategy_returns: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_weights: dict[str, list[pd.Series]] = {}
    strategy_predictions: dict[str, list[pd.Series]] = {}
    benchmark_records: list[tuple[pd.Timestamp, float]] = []
    equal_weight_records: list[tuple[pd.Timestamp, float]] = []

    for end_idx in range(window_size - 1, len(X_full)):
        X_window = X_full.iloc[end_idx - window_size + 1:end_idx + 1]
        realized_date = pd.Timestamp(target_dates[end_idx])
        realized_returns = Y_targets.iloc[end_idx]

        regime_state = compute_window_regime_state(X_window, regime_count=regime_count)
        R_window = regime_state["regimes"]
        P_window = regime_state["probabilities"]
        E_window = compute_transition_matrix(R_window, n_regimes)
        current_probs = renormalize_probabilities(P_window.iloc[-1].to_numpy())
        p_next = next_regime_probs(current_probs, E_window.to_numpy())
        predicted_regime = int(np.argmax(p_next))

        random_regimes, random_probs = compute_random_regime_state(X_window.index, n_regimes, rng)
        E_random = compute_transition_matrix(random_regimes, n_regimes)
        p_next_random = next_regime_probs(random_probs.iloc[-1].to_numpy(), E_random.to_numpy())
        predicted_regime_random = int(np.argmax(p_next_random))

        X_train = X_window.iloc[:-1]
        Y_train = Y_targets.loc[X_train.index]
        R_train = R_window.iloc[:-1]
        R_random_train = random_regimes.iloc[:-1]
        X_current = X_window.iloc[-1]

        family_scores = {
            "naive": standardize_scores(forecast_naive_sharpe(Y_train, R_train, p_next), Y_targets.columns),
            "naive_random": standardize_scores(forecast_naive_sharpe(Y_train, R_random_train, p_next_random), Y_targets.columns),
            "black_litterman": standardize_scores(forecast_black_litterman_scores(Y_train, R_train, p_next), Y_targets.columns),
            "mvo": standardize_scores(forecast_mvo_scores(Y_train), Y_targets.columns),
            "ridge": standardize_scores(pd.Series(predict_ridge(train_ridge_models(X_train, Y_train, R_train, n_regimes, alpha=ridge_alpha), X_current, p_next, n_regimes), index=Y_targets.columns), Y_targets.columns),
            "ridge_random": standardize_scores(pd.Series(predict_ridge(train_ridge_models(X_train, Y_train, R_random_train, n_regimes, alpha=ridge_alpha), X_current, p_next_random, n_regimes), index=Y_targets.columns), Y_targets.columns),
        }

        predicted_regimes = {
            "naive": predicted_regime,
            "naive_random": predicted_regime_random,
            "black_litterman": predicted_regime,
            "mvo": predicted_regime,
            "ridge": predicted_regime,
            "ridge_random": predicted_regime_random,
        }

        for family in MODEL_FAMILIES:
            scores = family_scores[family]
            for l_value in l_values:
                for mode in sizing_modes:
                    strategy = f"{family}_{mode}_l{l_value}"
                    weights = position_weights(scores, mode, l_value, predicted_regime=predicted_regimes[family])
                    strategy_returns.setdefault(strategy, []).append((realized_date, float(np.dot(weights.to_numpy(), realized_returns.to_numpy()))))
                    strategy_weights.setdefault(strategy, []).append(pd.Series(weights, name=realized_date))
                    strategy_predictions.setdefault(strategy, []).append(pd.Series(scores, name=realized_date))

        benchmark_records.append((realized_date, float(realized_returns[benchmark_col])))
        equal_weight_records.append((realized_date, float(realized_returns.mean())))

    portfolio_returns = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_returns.items()}
    portfolio_returns["SPY_buy_and_hold"] = pd.Series(dict(benchmark_records), name="SPY_buy_and_hold").sort_index()
    portfolio_returns["equal_weight_benchmark"] = pd.Series(dict(equal_weight_records), name="equal_weight_benchmark").sort_index()
    portfolio_returns_df = pd.DataFrame(portfolio_returns).sort_index()
    portfolio_returns_df.index.name = "date"

    panel_index = pd.Index(portfolio_returns_df.index, name="date")
    weights_panel = {strategy: pd.DataFrame(records, index=panel_index) for strategy, records in strategy_weights.items()}
    predictions_panel = {strategy: pd.DataFrame(records, index=panel_index) for strategy, records in strategy_predictions.items()}

    metrics_table = build_metrics_table(portfolio_returns_df)
    return {
        "portfolio_returns": portfolio_returns_df,
        "weights": flatten_panel(weights_panel),
        "predictions": flatten_panel(predictions_panel),
        "metrics_table": metrics_table,
        "metrics_json": metrics_table.set_index("strategy").to_dict(orient="index"),
    }
