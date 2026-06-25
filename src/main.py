from __future__ import annotations

try:
    from .pipeline import build_parser, run_pipeline
except ImportError:  # pragma: no cover
    from pipeline import build_parser, run_pipeline


def main() -> None:
    args = build_parser().parse_args()
    results = run_pipeline(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        kernel=args.kernel,
        kernel_components=args.kernel_components,
        gamma=args.gamma,
        degree=args.degree,
        coef0=args.coef0,
        regime_count=args.regime_count,
        trim_rows=args.trim_rows,
        plot=not args.no_plots,
        plot_format=args.plot_format,
        show_plots=args.show_plots,
        ridge_alpha=args.ridge_alpha,
        etf_tickers=args.etf_tickers,
        download_etfs=args.download_etfs,
        backtest=args.backtest,
        window_size=args.window_size,
        transaction_cost_bps=args.transaction_cost_bps,
        forecast_mode=args.forecast_mode,
        enable_cash_asset=bool(args.cash_asset and not args.no_cash_asset),
        fixed_overlay_exposure=args.fixed_overlay_exposure,
        cash_ticker=args.cash_ticker,
        enable_dynamic_risk_overlay=args.dynamic_risk_overlay,
        overlay_lookback_months=args.overlay_lookback_months,
        overlay_soft_drawdown=args.overlay_soft_drawdown,
        overlay_hard_drawdown=args.overlay_hard_drawdown,
        overlay_soft_exposure=args.overlay_soft_exposure,
        overlay_hard_exposure=args.overlay_hard_exposure,
        random_state=args.random_state,
    )

    print(f"Input CSV: {results['input_csv']}")
    print(f"Processed data saved to: {results['processed_path']}")
    print(f"Reduced PCA data saved to: {results['reduced_path']}")
    print(f"Regime assignments saved to: {results['regimes_path']}")
    if results["asset_returns"] is not None:
        print(f"ETF returns path: {results['backtest_paths']['etf_returns']}")
    if results["backtest"] is not None:
        print(f"Net portfolio returns saved to: {results['backtest_paths']['portfolio_returns']}")
        print(f"Gross portfolio returns saved to: {results['backtest_paths']['gross_portfolio_returns']}")
        print(f"Turnover saved to: {results['backtest_paths']['turnover']}")
        print(f"Transaction costs saved to: {results['backtest_paths']['transaction_costs']}")
        print(f"Overlay exposures saved to: {results['backtest_paths']['overlay_exposures']}")
        print(f"Overlay recent drawdowns saved to: {results['backtest_paths']['overlay_recent_drawdowns']}")
        print(f"Weights saved to: {results['backtest_paths']['weights']}")
        print(f"Predictions saved to: {results['backtest_paths']['predictions']}")
        print(f"Metrics saved to: {results['backtest_paths']['metrics']}")
        print(f"Forecast mode: {results['backtest']['forecast_mode']}")
        print(f"Forecast modes evaluated: {', '.join(results['backtest']['forecast_modes_evaluated'])}")
        print(f"Cash asset: {results['backtest'].get('cash_ticker')}")
        print(f"Fixed overlay exposure: {results['backtest'].get('fixed_overlay_exposure')}")
        print(f"Dynamic risk overlay: {results['backtest'].get('dynamic_risk_overlay_enabled')}")
        print(f"Dynamic overlay rules: {results['backtest'].get('dynamic_risk_overlay_rules')}")
        print(f"Metrics table saved to: {results['backtest_paths']['metrics_table']}")
        if results['backtest']['forecast_mode'] == 'both':
            print(f"Forecast comparison graph saved to: {results['backtest_paths']['forecast_mode_comparison']}")
    if results["pca_plot_path"] is not None:
        print(f"PCA regime plot saved to: {results['pca_plot_path']}")
    if results["timeline_plot_path"] is not None:
        print(f"Timeline regime plot saved to: {results['timeline_plot_path']}")
    print(f"PCA components selected: {results['pca_components']}")


if __name__ == "__main__":
    main()
