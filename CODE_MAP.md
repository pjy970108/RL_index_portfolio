# Code Map

This file maps the submitted code surface to the paper workflow.

| Stage | Location | Role |
|---|---|---|
| Data construction | `data_pipeline/` | Builds local price, feature, and tensor artifacts. |
| Final model | `modeling/grpo_sharpe/` | GRPO with Sharpe reward and min-max normalized group-return advantage. |
| PPO benchmark | `modeling/ppo_benchmark/` | PPO comparison model. |
| SAC benchmark | `modeling/sac_benchmark/` | SAC comparison model. |
| Backtest utilities | `backtest/` | Portfolio strategy returns and performance metrics. |
| Paper figures | `result/figures/paper_figures/` | Minimal retained thesis figure assets. |
| Run scripts | `scripts/` | Root-level training and evaluation entry points. |
| Data contract | `data/README.md` | Required local files excluded from Git. |

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

## Excluded From Submission Surface

The submitted branch excludes local datasets, checkpoints, W&B artifacts,
large intermediate tables, old external-model experiments, and broken legacy
min-max variant scripts. The original code history is preserved in the backup
branch.
