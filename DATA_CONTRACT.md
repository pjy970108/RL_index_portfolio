# Data Contract

This project follows the data construction described in the paper rather than a
filename-version convention.

## Periods

| Split | Period | Purpose |
|---|---|---|
| Raw data | 2002.02-2024.12 | Full price history for the asset universe |
| Lookback | Previous 252 trading days | Strategy and feature estimation window |
| Train | 2003.02.19-2013.12.31 | Reinforcement-learning training |
| Validation | 2014.01.01-2018.12.31 | Early stopping and model selection |
| Test | 2019.01.01-2024.12.31 | Final out-of-sample evaluation |

The raw universe contains 30 assets: 18 global equity indices and 12 commodity
futures. The February 2002 observations provide the initial 252-trading-day
lookback, so the first effective model state starts on February 19, 2003.

## Feature Pipeline

The model does not use raw asset prices directly as the RL state.

```text
Raw price data
        |
Daily returns
        |
252-trading-day rolling window
        |
Strategy portfolios: RP / MS / MV / PAA
        |
13 portfolio-level features per strategy
        |
Train-set min-max scaling
        |
52-dimensional RL state
```

For each of the four strategy portfolios, the pipeline derives 13 features:

| Group | Features |
|---|---|
| Return | 1-day return, cumulative return, 1-month return, 3-month return, 6-month return, 12-month return, average return |
| Risk and risk-adjusted | 20-day volatility, 252-day volatility, Sharpe ratio, Sortino ratio, Calmar ratio |
| Momentum | Weighted momentum score |

The momentum score is:

```text
Momentum = (12 * r_1m + 4 * r_3m + 2 * r_6m + r_12m) / 19
```

Each feature is min-max scaled to the [0, 1] range:

```text
x_scaled = (x - x_min) / (x_max - x_min)
```

The scaling parameters are estimated from the training set only and then reused
for validation and test data to avoid data leakage.

## Experimental Conditions

- No short selling: weights are non-negative.
- Fully invested portfolios: weights sum to one.
- Risk-free rate is zero.
- Rebalancing occurs every 20 trading days.
- Rebalancing uses opening prices.
- The default transaction cost is 30 bps multiplied by portfolio turnover.
