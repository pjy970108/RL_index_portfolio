# Evaluation Protocol

The final evaluation is performed on the complete out-of-sample test period:

| Split | Period | Role |
|---|---|---|
| Validation | 2014.01.01-2018.12.31 | Early stopping and model selection only |
| Test | 2019.01.01-2024.12.31 | Final out-of-sample evaluation |

The selected checkpoint is applied to the test period without retraining or
test-period tuning.

## Comparison Set

The final GRPO model is compared against:

- RL baselines: PPO and SAC
- Traditional strategy portfolios: RP, MV, MS, and PAA
- Simple benchmarks: ES and EA

ES is the equal-weight mixture of the four strategy portfolios. EA is the
equal-weight allocation across all assets.

## Reported Metrics

The main test-period table reports:

- Annualized return
- Sharpe ratio
- Sortino ratio
- Calmar ratio
- Maximum drawdown
- Profit factor

RL model results are reported as the average across five random seeds.

## Statistical Tests

Statistical significance is part of the final evaluation surface:

- Monthly mean return: Newey-West HAC t-test on test-period monthly returns.
- Sharpe ratio difference: Ledoit-Wolf style HAC Sharpe ratio difference test.

The reusable implementation is in `evaluation/statistical_tests.py`.

## Distribution And Downside Analysis

The monthly return distribution analysis reports mean, median, quartiles, IQR,
skewness, kurtosis, and outliers. Downside risk is evaluated with the bottom
25% monthly returns, including mean, standard deviation, minimum, maximum,
skewness, and kurtosis.

Profit/loss analysis counts profitable and loss months and compares their
average magnitudes.

## Market Regimes

The test period is also evaluated by market regime:

| Regime | Period |
|---|---|
| Pre-Pandemic | 2019.01-2020.02 |
| Pandemic / Inflation Surge | 2020.03-2022.06 |
| Global Disinflation | 2022.07-2024.12 |

Each regime uses the same performance metrics as the full test-period table.

## Robustness And Sensitivity

The retained robustness checks are:

- Asset exclusion test with a reduced 24-asset universe and no retraining.
- Transaction cost sensitivity at 0, 10, 20, 30, 40, and 50 bps.
- GRPO action-sampling sensitivity using 8, 16, and 32 sampled actions.

The paper figures retained under `result/figures/paper_figures/` correspond to
the final reporting layer; intermediate notebooks and large result tables are
not tracked in Git.
