from __future__ import annotations

import numpy as np
import pandas as pd

from .analytics import build_metrics_table
from .config import (
    DEFAULT_CASH_TICKER,
    DEFAULT_ENABLE_CASH_ASSET,
    DEFAULT_ENABLE_DYNAMIC_RISK_OVERLAY,
    DEFAULT_FIXED_OVERLAY_EXPOSURE,
    DEFAULT_FORECAST_MODE,
    DEFAULT_L_VALUES,
    DEFAULT_OVERLAY_HARD_DRAWDOWN,
    DEFAULT_OVERLAY_HARD_EXPOSURE,
    DEFAULT_OVERLAY_GOOD_PROBABILITY_THRESHOLD,
    DEFAULT_OVERLAY_GOOD_REGIME_COUNT,
    DEFAULT_OVERLAY_LOOKBACK_MONTHS,
    DEFAULT_OVERLAY_SOFT_DRAWDOWN,
    DEFAULT_OVERLAY_SOFT_EXPOSURE,
    DEFAULT_SIZING_MODES,
    DEFAULT_TRANSACTION_COST_BPS,
    MODEL_FAMILIES,
    RANDOM_SEED,
)
from .forecasting import (
    compute_random_regime_state,
    forecast_black_litterman_scores,
    forecast_mvo_scores,
    forecast_naive_sharpe,
    predict_ridge,
    regime_weights,
    train_ridge_models,
)
from .portfolio import add_cash_asset, apply_fixed_risk_overlay, evolve_weights, position_weights, standardize_scores, traded_notional, transaction_cost
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



def _regime_label_to_int(label: object) -> int:
    """Convert labels such as 0 or 'regime_prob_0' to integer regime ids."""
    if isinstance(label, (int, np.integer)):
        return int(label)
    text = str(label)
    if text.startswith("regime_prob_"):
        return int(text.split("regime_prob_", 1)[1])
    return int(text)

def rank_good_regimes_by_equal_weight_returns(
    returns: pd.DataFrame,
    regime_probabilities: pd.DataFrame,
    risky_asset_columns: list[str],
    good_regime_count: int = DEFAULT_OVERLAY_GOOD_REGIME_COUNT,
) -> tuple[set[int], pd.Series]:
    """Rank regimes using past equal-weight ETF returns.

    The ranking is computed inside the current expanding training window only.
    For every historical month we first compute the equal-weight average return
    across all risky ETFs. Then, for each regime, we compute the probability-
    weighted average of that equal-weight return. This works with soft regime
    memberships and avoids relying on arbitrary cluster labels.

    Returns
    -------
    good_regimes:
        Set containing the best `good_regime_count` regime labels.
    regime_scores:
        Probability-weighted equal-weight return score for every regime.
    """
    if regime_probabilities.empty:
        return set(), pd.Series(dtype=float)

    aligned_returns = returns.reindex(regime_probabilities.index)
    available_assets = [col for col in risky_asset_columns if col in aligned_returns.columns]
    if not available_assets:
        raise ValueError("No risky ETF return columns are available for regime ranking.")

    equal_weight_market_return = aligned_returns[available_assets].astype(float).mean(axis=1)
    regime_scores: dict[int, float] = {}

    for regime in regime_probabilities.columns:
        regime_id = _regime_label_to_int(regime)
        weights = regime_probabilities[regime].astype(float).reindex(equal_weight_market_return.index).fillna(0.0)
        valid = equal_weight_market_return.notna() & weights.notna()
        denom = float(weights[valid].sum())
        if denom <= 0:
            regime_scores[regime_id] = np.nan
        else:
            regime_scores[regime_id] = float((equal_weight_market_return[valid] * weights[valid]).sum() / denom)

    scores = pd.Series(regime_scores, dtype=float).sort_index()
    valid_scores = scores.dropna()
    if valid_scores.empty:
        return set(), scores

    n_good = max(1, min(int(good_regime_count), len(valid_scores)))
    good_regimes = set(int(regime) for regime in valid_scores.sort_values(ascending=False).head(n_good).index)
    return good_regimes, scores


def regime_good_probability(regime_probabilities: pd.Series | np.ndarray, good_regimes: set[int]) -> float:
    """Return the probability mass assigned to the current good-regime group."""
    if not good_regimes:
        return 0.0
    probs = pd.Series(regime_probabilities, dtype=float)
    return float(probs.reindex(sorted(good_regimes)).fillna(0.0).sum())


def recent_drawdown_overlay_exposure(
    return_records: list[tuple[pd.Timestamp, float]],
    lookback_months: int = DEFAULT_OVERLAY_LOOKBACK_MONTHS,
    soft_drawdown: float = DEFAULT_OVERLAY_SOFT_DRAWDOWN,
    hard_drawdown: float = DEFAULT_OVERLAY_HARD_DRAWDOWN,
    soft_exposure: float = DEFAULT_OVERLAY_SOFT_EXPOSURE,
    hard_exposure: float = DEFAULT_OVERLAY_HARD_EXPOSURE,
    next_regime_probabilities: pd.Series | np.ndarray | None = None,
    good_regimes: set[int] | None = None,
    good_probability_threshold: float = DEFAULT_OVERLAY_GOOD_PROBABILITY_THRESHOLD,
) -> tuple[float, float, float, str]:
    """Return overlay decision using past drawdown plus regime re-entry.

    The base protection rule uses only the last `lookback_months` already-
    realized net strategy returns, so the current month is not used.

    If recent drawdown is not bad, exposure is 100%.
    If recent drawdown is bad but the probability of next month belonging to
    the good-regime group is at least `good_probability_threshold`, exposure
    returns immediately to 100%. This is the re-entry rule that avoids staying
    too long in cash during recoveries.
    """
    lookback_months = int(lookback_months)
    good_probability = 0.0
    if next_regime_probabilities is not None and good_regimes:
        good_probability = regime_good_probability(next_regime_probabilities, good_regimes)

    if lookback_months <= 0 or len(return_records) < lookback_months:
        return 1.0, 0.0, good_probability, "insufficient_history"

    recent = pd.Series([ret for _, ret in return_records[-lookback_months:]], dtype=float)
    equity = pd.concat([pd.Series([1.0]), (1.0 + recent).cumprod()], ignore_index=True)
    drawdown = equity / equity.cummax() - 1.0
    recent_max_drawdown = float(drawdown.min())

    if recent_max_drawdown > float(soft_drawdown):
        return 1.0, recent_max_drawdown, good_probability, "normal_drawdown"

    if good_probability >= float(good_probability_threshold):
        return 1.0, recent_max_drawdown, good_probability, "good_regime_reentry"

    if recent_max_drawdown <= float(hard_drawdown):
        return float(hard_exposure), recent_max_drawdown, good_probability, "hard_drawdown"

    return float(soft_exposure), recent_max_drawdown, good_probability, "soft_drawdown"


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
    enable_cash_asset: bool = DEFAULT_ENABLE_CASH_ASSET,
    fixed_overlay_exposure: float = DEFAULT_FIXED_OVERLAY_EXPOSURE,
    cash_ticker: str = DEFAULT_CASH_TICKER,
    enable_dynamic_risk_overlay: bool = DEFAULT_ENABLE_DYNAMIC_RISK_OVERLAY,
    overlay_lookback_months: int = DEFAULT_OVERLAY_LOOKBACK_MONTHS,
    overlay_soft_drawdown: float = DEFAULT_OVERLAY_SOFT_DRAWDOWN,
    overlay_hard_drawdown: float = DEFAULT_OVERLAY_HARD_DRAWDOWN,
    overlay_soft_exposure: float = DEFAULT_OVERLAY_SOFT_EXPOSURE,
    overlay_hard_exposure: float = DEFAULT_OVERLAY_HARD_EXPOSURE,
    overlay_good_probability_threshold: float = DEFAULT_OVERLAY_GOOD_PROBABILITY_THRESHOLD,
    overlay_good_regime_count: int = DEFAULT_OVERLAY_GOOD_REGIME_COUNT,
) -> dict[str, object]:
    if len(X_full) < window_size:
        raise ValueError(f"Not enough aligned observations. Need at least {window_size} rows to start. Found {len(X_full)} rows.")

    n_regimes = regime_count + 1
    risky_asset_columns = list(Y_targets.columns)
    dynamic_overlay_enabled = bool(enable_dynamic_risk_overlay)
    fixed_overlay_enabled = float(fixed_overlay_exposure) < 1.0
    cash_enabled = bool(enable_cash_asset or dynamic_overlay_enabled or fixed_overlay_enabled)
    if cash_enabled and cash_ticker not in Y_targets.columns:
        Y_targets = Y_targets.copy()
        Y_targets[cash_ticker] = 0.0
    asset_columns = list(Y_targets.columns)
    benchmark_col = "SPY" if "SPY" in risky_asset_columns else risky_asset_columns[0]
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
    strategy_overlay_exposures: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_overlay_drawdowns: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_overlay_good_probabilities: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    strategy_overlay_actions: dict[str, list[tuple[pd.Timestamp, str]]] = {}

    previous_strategy_weights: dict[str, pd.Series] = {}
    previous_realized_returns: pd.Series | None = None

    benchmark_templates = {
        "SPY_buy_and_hold": pd.Series(0.0, index=asset_columns, dtype=float),
        "equal_weight_benchmark": pd.Series(0.0, index=asset_columns, dtype=float),
    }
    benchmark_templates["SPY_buy_and_hold"].loc[benchmark_col] = 1.0
    benchmark_templates["equal_weight_benchmark"].loc[risky_asset_columns] = 1.0 / len(risky_asset_columns)
    benchmark_gross_returns = {name: [] for name in benchmark_templates}
    benchmark_net_returns = {name: [] for name in benchmark_templates}
    benchmark_turnover = {name: [] for name in benchmark_templates}
    benchmark_transaction_costs = {name: [] for name in benchmark_templates}
    previous_benchmark_weights: dict[str, pd.Series] = {}


    for end_idx in range(window_size - 1, len(X_full)):
        X_window = X_full.iloc[: end_idx + 1]
        realized_date = pd.Timestamp(target_dates[end_idx])
        realized_returns = Y_targets.iloc[end_idx].astype(float)

        regime_state = compute_window_regime_state(X_window, regime_count=regime_count)
        R_window = regime_state["regimes"]
        P_window = regime_state["probabilities"]
        # Use soft regime probabilities for expected transition counts.
        # The previous hard-label version ignored most of the soft-clustering information here.
        E_window = compute_transition_matrix(P_window, n_regimes)
        current_probs = renormalize_probabilities(P_window.iloc[-1].to_numpy())
        p_next = next_regime_probs(current_probs, E_window.to_numpy())

        random_regimes, random_probs = compute_random_regime_state(X_window.index, n_regimes, rng)
        E_random = compute_transition_matrix(random_regimes, n_regimes)
        p_next_random = next_regime_probs(random_probs.iloc[-1].to_numpy(), E_random.to_numpy())

        X_train = X_window.iloc[:-1]
        Y_train = Y_targets.loc[X_train.index]

        half_life = 48  # months, tune this
        ages = np.arange(len(Y_train) - 1, -1, -1)
        time_weights = 0.5 ** (ages / half_life)
        time_weights = time_weights / time_weights.sum()

        R_train = R_window.iloc[:-1]
        P_train = P_window.iloc[:-1]
        R_random_train = random_regimes.iloc[:-1]
        P_random_train = random_probs.iloc[:-1]

        # Regime-aware re-entry ranking for the dynamic overlay.
        # This is recomputed every rebalance using only the expanding-window past.
        # Regimes are ranked by the probability-weighted equal-weight average
        # return across all risky ETFs during that regime.
        overlay_good_regimes, overlay_regime_scores = rank_good_regimes_by_equal_weight_returns(
            Y_train,
            P_train,
            risky_asset_columns,
            good_regime_count=overlay_good_regime_count,
        )
        overlay_random_good_regimes, overlay_random_regime_scores = rank_good_regimes_by_equal_weight_returns(
            Y_train,
            P_random_train,
            risky_asset_columns,
            good_regime_count=overlay_good_regime_count,
        )

        X_current = X_window.iloc[-1]
        ridge_models = train_ridge_models(X_train, Y_train, R_train, n_regimes, alpha=ridge_alpha, sample_weights=time_weights)
        ridge_random_models = train_ridge_models(X_train, Y_train, R_random_train, n_regimes, alpha=ridge_alpha, sample_weights=time_weights)

        forecast_payloads: dict[str, tuple[dict[str, pd.Series], dict[str, pd.Series]]] = {}
        for active_forecast_mode in forecast_modes:
            family_scores = {
                "naive": standardize_scores(
                    forecast_naive_sharpe(
                        Y_train,
                        R_train,
                        p_next,
                        forecast_mode=active_forecast_mode,
                        sample_weights=time_weights,
                        regime_probabilities=P_train,
                    ),
                    Y_targets.columns,
                ),
                "naive_random": standardize_scores(
                    forecast_naive_sharpe(
                        Y_train,
                        R_random_train,
                        p_next_random,
                        forecast_mode=active_forecast_mode,
                        sample_weights=time_weights,
                        regime_probabilities=P_random_train,
                    ),
                    Y_targets.columns,
                ),
                "black_litterman": standardize_scores(
                    forecast_black_litterman_scores(
                        Y_train,
                        R_train,
                        p_next,
                        forecast_mode=active_forecast_mode,
                        sample_weights=time_weights,
                        regime_probabilities=P_train,
                    ),
                    Y_targets.columns,
                ),
                "mvo": standardize_scores(
                    forecast_mvo_scores(
                        Y_train,
                        R_train,
                        p_next,
                        forecast_mode=active_forecast_mode,
                        sample_weights=time_weights,
                        regime_probabilities=P_train,
                    ),
                    Y_targets.columns,
                ),
                "ridge": standardize_scores(pd.Series(predict_ridge(ridge_models, X_current, p_next, n_regimes, forecast_mode=active_forecast_mode), index=Y_targets.columns), Y_targets.columns),
                "ridge_random": standardize_scores(pd.Series(predict_ridge(ridge_random_models, X_current, p_next_random, n_regimes, forecast_mode=active_forecast_mode), index=Y_targets.columns), Y_targets.columns),
            }
            if cash_enabled:
                family_scores = {
                    family_name: add_cash_asset(scores, cash_ticker=cash_ticker, cash_score=0.0)
                    for family_name, scores in family_scores.items()
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
                        if cash_enabled:
                            weights = weights.reindex(asset_columns).fillna(0.0).astype(float)

                        overlay_exposure = 1.0
                        overlay_recent_drawdown = 0.0
                        overlay_good_probability = 0.0
                        overlay_action = "no_overlay"
                        if dynamic_overlay_enabled:
                            family_good_regimes = overlay_random_good_regimes if MODEL_FAMILIES[family].get("random_regimes") else overlay_good_regimes
                            overlay_exposure, overlay_recent_drawdown, overlay_good_probability, overlay_action = recent_drawdown_overlay_exposure(
                                strategy_net_returns.get(strategy, []),
                                lookback_months=overlay_lookback_months,
                                soft_drawdown=overlay_soft_drawdown,
                                hard_drawdown=overlay_hard_drawdown,
                                soft_exposure=overlay_soft_exposure,
                                hard_exposure=overlay_hard_exposure,
                                next_regime_probabilities=regime_probability_sets[family],
                                good_regimes=family_good_regimes,
                                good_probability_threshold=overlay_good_probability_threshold,
                            )
                            weights = apply_fixed_risk_overlay(weights, overlay_exposure, cash_ticker=cash_ticker)
                        elif fixed_overlay_exposure < 1.0:
                            overlay_exposure = float(fixed_overlay_exposure)
                            overlay_action = "fixed_overlay"
                            weights = apply_fixed_risk_overlay(weights, fixed_overlay_exposure, cash_ticker=cash_ticker)

                        weights = weights.reindex(asset_columns).fillna(0.0).astype(float)
                        start_weights = _starting_weights(previous_strategy_weights.get(strategy), previous_realized_returns, asset_columns)
                        turnover = traded_notional(start_weights, weights, cash_ticker=cash_ticker if cash_enabled else None)
                        trading_cost = transaction_cost(start_weights, weights, cost_rate, cash_ticker=cash_ticker if cash_enabled else None)
                        gross_return = float(np.dot(weights.to_numpy(), realized_returns.to_numpy()))
                        net_return = gross_return - trading_cost

                        strategy_gross_returns.setdefault(strategy, []).append((realized_date, gross_return))
                        strategy_net_returns.setdefault(strategy, []).append((realized_date, net_return))
                        strategy_turnover.setdefault(strategy, []).append((realized_date, turnover))
                        strategy_transaction_costs.setdefault(strategy, []).append((realized_date, trading_cost))
                        strategy_overlay_exposures.setdefault(strategy, []).append((realized_date, overlay_exposure))
                        strategy_overlay_drawdowns.setdefault(strategy, []).append((realized_date, overlay_recent_drawdown))
                        strategy_overlay_good_probabilities.setdefault(strategy, []).append((realized_date, overlay_good_probability))
                        strategy_overlay_actions.setdefault(strategy, []).append((realized_date, overlay_action))
                        strategy_weights.setdefault(strategy, []).append(pd.Series(weights, name=realized_date))
                        strategy_predictions.setdefault(strategy, []).append(pd.Series(scores, name=realized_date))
                        previous_strategy_weights[strategy] = weights

        for benchmark_name, target_weights in benchmark_templates.items():
            start_weights = _starting_weights(previous_benchmark_weights.get(benchmark_name), previous_realized_returns, asset_columns)
            turnover = traded_notional(start_weights, target_weights, cash_ticker=cash_ticker if cash_enabled else None)
            trading_cost = transaction_cost(start_weights, target_weights, cost_rate, cash_ticker=cash_ticker if cash_enabled else None)
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

    overlay_exposure_records = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_overlay_exposures.items()}
    overlay_exposure_df = pd.DataFrame(overlay_exposure_records).sort_index()
    overlay_exposure_df.index.name = "date"

    overlay_drawdown_records = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_overlay_drawdowns.items()}
    overlay_drawdown_df = pd.DataFrame(overlay_drawdown_records).sort_index()
    overlay_drawdown_df.index.name = "date"

    overlay_good_probability_records = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_overlay_good_probabilities.items()}
    overlay_good_probability_df = pd.DataFrame(overlay_good_probability_records).sort_index()
    overlay_good_probability_df.index.name = "date"

    overlay_action_records = {strategy: pd.Series(dict(records), name=strategy).sort_index() for strategy, records in strategy_overlay_actions.items()}
    overlay_action_df = pd.DataFrame(overlay_action_records).sort_index()
    overlay_action_df.index.name = "date"

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
        "overlay_exposures": overlay_exposure_df,
        "overlay_recent_drawdowns": overlay_drawdown_df,
        "overlay_good_probabilities": overlay_good_probability_df,
        "overlay_actions": overlay_action_df,
        "weights": flatten_panel(weights_panel),
        "predictions": flatten_panel(predictions_panel),
        "metrics_table": metrics_table,
        "metrics_json": metrics_table.set_index("strategy").to_dict(orient="index"),
        "transaction_cost_bps": float(transaction_cost_bps),
        "forecast_mode": forecast_mode,
        "forecast_modes_evaluated": list(forecast_modes),
        "cash_ticker": cash_ticker if cash_enabled else None,
        "fixed_overlay_exposure": float(fixed_overlay_exposure),
        "dynamic_risk_overlay_enabled": dynamic_overlay_enabled,
        "dynamic_risk_overlay_rules": {
            "lookback_months": int(overlay_lookback_months),
            "soft_drawdown": float(overlay_soft_drawdown),
            "soft_exposure": float(overlay_soft_exposure),
            "hard_drawdown": float(overlay_hard_drawdown),
            "hard_exposure": float(overlay_hard_exposure),
            "good_probability_threshold": float(overlay_good_probability_threshold),
            "good_regime_count": int(overlay_good_regime_count),
            "good_regime_ranking": "expanding_window_equal_weight_average_etf_return",
        },
    }
