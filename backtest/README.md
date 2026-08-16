# Backtest Utilities

This folder keeps the backtesting code used by the final training and
out-of-sample evaluation paths.

| File | Role |
|---|---|
| `dynamic_portfolio.py` | Daily return version of RP, MV, MS, PAA, and equal-weight portfolio construction. |
| `dynamic_portfolio_monthly.py` | Monthly/20-trading-day aggregated version used by final evaluation. |
| `backtesting_all_asset.py` | Rolling daily strategy-return wrapper used by the training environments. |
| `backtesting_all_asset_monthly.py` | Rolling monthly strategy-return wrapper used by evaluation and validation scoring. |
| `backtesting.py` | Compatibility wrapper for the remaining sub-environment code path. |
| `eval_metric.py` | Shared performance metrics used inside training/evaluation loops. |

The final paper evaluation should use the monthly all-asset path:

```text
dynamic_portfolio_monthly.py
        |
backtesting_all_asset_monthly.py
        |
modeling/*/run.py and modeling/*/*_monthly.py
```

The strategy constraints follow the thesis setup: long-only weights,
fully-invested portfolios, risk-free rate set by config, and transaction costs
applied from portfolio turnover.
