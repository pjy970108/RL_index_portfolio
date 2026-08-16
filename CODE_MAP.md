# Code Map

This file maps the submitted code surface to the paper workflow.

| Stage | Location | Role |
|---|---|---|
| Data construction | `data_pipeline/` | Final notebooks for price preparation, strategy features, and RL state tensors. |
| Final model | `modeling/grpo_sharpe/` | GRPO with Sharpe reward and min-max normalized group-return advantage. |
| PPO benchmark | `modeling/ppo_benchmark/` | PPO comparison model. |
| SAC benchmark | `modeling/sac_benchmark/` | SAC comparison model. |
| Backtest utilities | `backtest/` | Final daily/monthly strategy-return wrappers and performance metrics. |
| Evaluation | `evaluation/` | Final OOS evaluation protocol, performance summaries, and statistical tests. |
| Paper figures | `result/` | Minimal retained thesis figure assets. |
| Run scripts | `scripts/` | Root-level training and evaluation entry points. |
| Data contract | `DATA_CONTRACT.md` | Data periods, lookback window, feature definitions, and leakage boundary. |

## Data Pipeline

The submitted data pipeline is limited to the final model-input path:

- `data_pipeline/01_prepare_price_universe.ipynb`
- `data_pipeline/02_create_chronological_splits.ipynb`
- `data_pipeline/03_build_rl_state_tensor.ipynb`

These notebooks implement the paper path from raw prices to 52-dimensional RL
states: 252-trading-day lookback, four strategy portfolios, 13 features per
strategy, and train-set-only min-max scaling.

## Final GRPO Path

The final GRPO implementation is centered on:

- `modeling/grpo_sharpe/train_concat_monthly_1m_discrete_rebal_step_batch_sample_seed_change.py`
- `modeling/grpo_sharpe/grpo_monthly.py`
- `modeling/grpo_sharpe/enviroment.py`
- `modeling/grpo_sharpe/agent.py`
- `modeling/grpo_sharpe/network.py`
- `modeling/grpo_sharpe/config/train_config.yaml`
- `modeling/grpo_sharpe/config/test_config.yaml`

The environment returns Sharpe-based rewards through `reward_cond: sharpe`.
`grpo_monthly.py` then computes trajectory-level group returns and applies
min-max normalization to form the GRPO advantage.

## Benchmarks

PPO benchmark:

- `modeling/ppo_benchmark/train_5d_discrete_rebal_step_seed.py`
- `modeling/ppo_benchmark/run.py`
- `modeling/ppo_benchmark/agent.py`
- `modeling/ppo_benchmark/enviroment.py`
- `modeling/ppo_benchmark/config/train_config.yaml`
- `modeling/ppo_benchmark/config/test_config.yaml`

SAC benchmark:

- `modeling/sac_benchmark/train_5d_discrete_rebal_step_seed.py`
- `modeling/sac_benchmark/run.py`
- `modeling/sac_benchmark/agent.py`
- `modeling/sac_benchmark/enviroment.py`
- `modeling/sac_benchmark/config/train_config.yaml`
- `modeling/sac_benchmark/config/test_config.yaml`

## Evaluation

The final evaluation is documented and implemented through:

- `evaluation/README.md`
- `evaluation/statistical_tests.py`

The evaluation contract is the complete test window
`2019.01.01-2024.12.31`. Validation is used for early stopping and model
selection only. The statistical test utilities cover Newey-West HAC mean-return
tests, Ledoit-Wolf style Sharpe ratio difference tests, monthly distribution
summaries, downside 25% summaries, profit/loss summaries, and regime-level
metrics.

## Excluded From Submission Surface

The submitted branch excludes local datasets, checkpoints, W&B artifacts,
large intermediate tables, exploratory notebooks, earlier data-construction
variants, robustness-only asset-removal notebooks, and external-model
experiments.
