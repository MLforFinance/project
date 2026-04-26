from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import build_metrics_table
from .config import DEFAULT_FORECAST_MODE, DEFAULT_L_VALUES, DEFAULT_SIZING_MODES, DEFAULT_TRANSACTION_COST_BPS, MODEL_FAMILIES, RANDOM_SEED
from .forecasting import (
    compute_random_regime_state,
    forecast_black_litterman_scores,
    forecast_mvo_scores,
    forecast_naive_sharpe,
    predict_ridge,
    regime_weights,
    train_ridge_models,
)
from .portfolio import evolve_weights, position_weights, standardize_scores, traded_notional, transaction_cost
from .regime_pipeline import compute_transition_matrix, compute_window_regime_state, next_regime_probs, renormalize_probabilities
from .reporting import flatten_panel


def _starting_weights(
    previous_target_weights: pd.Series | None,
    previous_realized_returns: pd.Series | None,
    asset_columns: list[str],
) -> pd.Series:
    if previous_target_weights is None:
        return pd.Series(0.0, index=asset_columns, dtype=float)
    if previous_realized_returns is None:
        return previous_target_weights.reindex(asset_columns).fillna(0.0).astype(float)
    return evolve_weights(previous_target_weights.reindex(asset_columns).fillna(0.0).astype(float), previous_realized_returns.reindex(asset_columns).fillna(0.0).astype(float))


def _forecast_modes_to_run(forecast_mode: str) -> tuple[str, ...]:
    if forecast_mode == "both":
        return ("hard", "soft")
    return (forecast_mode,)


def _strategy_name(family: str, mode: str, l_value: int, forecast_mode: str, include_suffix: bool) -> str:
    base = f"{family}_{mode}_l{l_value}"
    return f"{base}__{forecast_mode}" if include_suffix else base


def run_walk_forward_backtest(
    X_full: pd.DataFrame,
    Y_targets: pd.DataFrame,
    target_dates: pd.Index,
    regime_count: int,
    window_size: int,
    ridge_alpha: float,
    l_values: tuple[int, ...] = DEFAULT_L_VALUES,
    sizing_modes: tuple[str, ...] = DEFAULT_SIZING_MODES,
    transaction_cost_bps: float = DEFAULT_TRANSACTION_COST_BPS,
    forecast_mode: str = DEFAULT_FORECAST_MODE,
) -> dict[str, object]:
    if len(X_full) < window_size:
        raise ValueError(f"Not enough aligned observations for a {window_size}-month window. Found {len(X_full)} rows.")

    n_regimes = regime_count + 1
    asset_columns = list(Y_targets.columns)
    benchmark_col = "SPY" if "SPY" in asset_columns else asset_columns[0]
    rng = np.random.default_rng(RANDOM_SEED)
    cost_rate = float(transaction_cost_bps) / 10000.0
    forecast_modes = _forecast_modes_to_run(forecast_mode)
    include_forecast_suffix = len(forecast_modes) > 1

    strategy_gross_returns: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_net_returns: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_weights: dict[str, list[pd.Series]] = {}
    strategy_predictions: dict[str, list[pd.Series]] = {}
    strategy_turnover: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_transaction_costs: dict[str, list[tuple[pd.Timestamp, float]]] = {}

    previous_strategy_weights: dict[str, pd.Series] = {}
    previous_realized_returns: pd.Series | None = None

    benchmark_templates = {
        "SPY_buy_and_hold": pd.Series(0.0, index=asset_columns, dtype=float),
        "equal_weight_benchmark": pd.Series(1.0 / len(asset_columns), index=asset_columns, dtype=float),
    }
    benchmark_templates["SPY_buy_and_hold"].loc[benchmark_col] = 1.0
    benchmark_gross_returns = {name: [] for name in benchmark_templates}
    benchmark_net_returns = {name: [] for name in benchmark_templates}
    benchmark_turnover = {name: [] for name in benchmark_templates}
    benchmark_transaction_costs = {name: [] for name in benchmark_templates}
    previous_benchmark_weights: dict[str, pd.Series] = {}

    for end_idx in range(window_size - 1, len(X_full)):
        X_window = X_full.iloc[end_idx - window_size + 1:end_idx + 1]
        realized_date = pd.Timestamp(target_dates[end_idx])
        realized_returns = Y_targets.iloc[end_idx].astype(float)

        regime_state = compute_window_regime_state(X_window, regime_count=regime_count)
        R_window = regime_state["regimes"]
        P_window = regime_state["probabilities"]
        E_window = compute_transition_matrix(R_window, n_regimes)
        current_probs = renormalize_probabilities(P_window.iloc[-1].to_numpy())
        p_next = next_regime_probs(current_probs, E_window.to_numpy())

        random_regimes, random_probs = compute_random_regime_state(X_window.index, n_regimes, rng)
        E_random = compute_transition_matrix(random_regimes, n_regimes)
        p_next_random = next_regime_probs(random_probs.iloc[-1].to_numpy(), E_random.to_numpy())

        X_train = X_window.iloc[:-1]
        Y_train = Y_targets.loc[X_train.index]
        R_train = R_window.iloc[:-1]
        R_random_train = random_regimes.iloc[:-1]
        X_current = X_window.iloc[-1]
        ridge_models = train_ridge_models(X_train, Y_train, R_train, n_regimes, alpha=ridge_alpha)
        ridge_random_models = train_ridge_models(X_train, Y_train, R_random_train, n_regimes, alpha=ridge_alpha)

        forecast_payloads: dict[str, tuple[dict[str, pd.Series], dict[str, pd.Series]]] = {}
        for active_forecast_mode in forecast_modes:
            family_scores = {
                "naive": standardize_scores(forecast_naive_sharpe(Y_train, R_train, p_next, forecast_mode=active_forecast_mode), Y_targets.columns),
                "naive_random": standardize_scores(forecast_naive_sharpe(Y_train, R_random_train, p_next_random, forecast_mode=active_forecast_mode), Y_targets.columns),
                "black_litterman": standardize_scores(forecast_black_litterman_scores(Y_train, R_train, p_next, forecast_mode=active_forecast_mode), Y_targets.columns),
                "mvo": standardize_scores(forecast_mvo_scores(Y_train), Y_targets.columns),
                "ridge": standardize_scores(pd.Series(predict_ridge(ridge_models, X_current, p_next, n_regimes, forecast_mode=active_forecast_mode), index=Y_targets.columns), Y_targets.columns),
                "ridge_random": standardize_scores(pd.Series(predict_ridge(ridge_random_models, X_current, p_next_random, n_regimes, forecast_mode=active_forecast_mode), index=Y_targets.columns), Y_targets.columns),
            }
            regime_probability_sets = {
                "naive": pd.Series(regime_weights(p_next, active_forecast_mode), index=range(n_regimes), dtype=float),
                "naive_random": pd.Series(regime_weights(p_next_random, active_forecast_mode), index=range(n_regimes), dtype=float),
                "black_litterman": pd.Series(regime_weights(p_next, active_forecast_mode), index=range(n_regimes), dtype=float),
                "mvo": pd.Series(regime_weights(p_next, active_forecast_mode), index=range(n_regimes), dtype=float),
                "ridge": pd.Series(regime_weights(p_next, active_forecast_mode), index=range(n_regimes), dtype=float),
                "ridge_random": pd.Series(regime_weights(p_next_random, active_forecast_mode), index=range(n_regimes), dtype=float),
            }
            forecast_payloads[active_forecast_mode] = (family_scores, regime_probability_sets)

        for active_forecast_mode, (family_scores, regime_probability_sets) in forecast_payloads.items():
            for family in MODEL_FAMILIES:
                scores = family_scores[family]
                for l_value in l_values:
                    for mode in sizing_modes:
                        strategy = _strategy_name(family, mode, l_value, active_forecast_mode, include_forecast_suffix)
                        weights = position_weights(scores, mode, l_value, regime_probs=regime_probability_sets[family]).astype(float)
                        start_weights = _starting_weights(previous_strategy_weights.get(strategy), previous_realized_returns, asset_columns)
                        turnover = traded_notional(start_weights, weights)
                        trading_cost = transaction_cost(start_weights, weights, cost_rate)
                        gross_return = float(np.dot(weights.to_numpy(), realized_returns.to_numpy()))
                        net_return = gross_return - trading_cost

                        strategy_gross_returns.setdefault(strategy, []).append((realized_date, gross_return))
                        strategy_net_returns.setdefault(strategy, []).append((realized_date, net_return))
                        strategy_turnover.setdefault(strategy, []).append((realized_date, turnover))
                        strategy_transaction_costs.setdefault(strategy, []).append((realized_date, trading_cost))
                        strategy_weights.setdefault(strategy, []).append(pd.Series(weights, name=realized_date))
                        strategy_predictions.setdefault(strategy, []).append(pd.Series(scores, name=realized_date))
                        previous_strategy_weights[strategy] = weights

        for benchmark_name, target_weights in benchmark_templates.items():
            start_weights = _starting_weights(previous_benchmark_weights.get(benchmark_name), previous_realized_returns, asset_columns)
            turnover = traded_notional(start_weights, target_weights)
            trading_cost = transaction_cost(start_weights, target_weights, cost_rate)
            gross_return = float(np.dot(target_weights.to_numpy(), realized_returns.to_numpy()))
            net_return = gross_return - trading_cost
            benchmark_gross_returns[benchmark_name].append((realized_date, gross_return))
            benchmark_net_returns[benchmark_name].append((realized_date, net_return))
            benchmark_turnover[benchmark_name].append((realized_date, turnover))
            benchmark_transaction_costs[benchmark_name].append((realized_date, trading_cost))
            previous_benchmark_weights[benchmark_name] = target_weights

        previous_realized_returns = realized_returns

    gross_returns = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_gross_returns.items()}
    gross_returns.update({strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in benchmark_gross_returns.items()})
    gross_returns_df = pd.DataFrame(gross_returns).sort_index()
    gross_returns_df.index.name = "date"

    net_returns = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_net_returns.items()}
    net_returns.update({strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in benchmark_net_returns.items()})
    portfolio_returns_df = pd.DataFrame(net_returns).sort_index()
    portfolio_returns_df.index.name = "date"

    turnover_records = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_turnover.items()}
    turnover_records.update({strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in benchmark_turnover.items()})
    turnover_df = pd.DataFrame(turnover_records).sort_index()
    turnover_df.index.name = "date"

    transaction_cost_records = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_transaction_costs.items()}
    transaction_cost_records.update({strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in benchmark_transaction_costs.items()})
    transaction_costs_df = pd.DataFrame(transaction_cost_records).sort_index()
    transaction_costs_df.index.name = "date"

    panel_index = pd.Index(portfolio_returns_df.index, name="date")
    weights_panel = {strategy: pd.DataFrame(records, index=panel_index) for strategy, records in strategy_weights.items()}
    predictions_panel = {strategy: pd.DataFrame(records, index=panel_index) for strategy, records in strategy_predictions.items()}

    metrics_table = build_metrics_table(
        portfolio_returns_df,
        gross_returns=gross_returns_df,
        turnover=turnover_df,
        transaction_costs=transaction_costs_df,
        default_forecast_mode=forecast_mode if forecast_mode != "both" else None,
    )
    return {
        "portfolio_returns": portfolio_returns_df,
        "gross_portfolio_returns": gross_returns_df,
        "turnover": turnover_df,
        "transaction_costs": transaction_costs_df,
        "weights": flatten_panel(weights_panel),
        "predictions": flatten_panel(predictions_panel),
        "metrics_table": metrics_table,
        "metrics_json": metrics_table.set_index("strategy").to_dict(orient="index"),
        "transaction_cost_bps": float(transaction_cost_bps),
        "forecast_mode": forecast_mode,
        "forecast_modes_evaluated": list(forecast_modes),
    }
