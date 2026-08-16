# Code Map

This file summarizes the paper-code layout.

| Stage | Location | Notes |
|---|---|---|
| Data preprocessing | `data_pipeline/` | Portfolio pricing, indicator construction, tensor generation, and EDA notebooks. |
| Main model | `modeling/grpo_sharpe/` | Final GRPO Sharpe implementation. |
| PPO benchmark | `modeling/ppo_benchmark/` | PPO comparison implementation and evaluation notebooks. |
| SAC benchmark | `modeling/sac_benchmark/` | SAC comparison implementation and evaluation notebooks. |
| Backtest utilities | `backtest/` | Dynamic portfolio strategies, partial backtests, and performance metrics. |
| Figure generation | `figure/` | Final figure and result analysis notebooks. |
| Result assets | `result/` | Generated PNG figures and analysis outputs. |
| Run scripts | `scripts/` | Shell entry points for training and evaluation. |
| Local data contract | `data/README.md` | Expected local files that are not committed to Git. |

## Main GRPO Files

- `modeling/grpo_sharpe/agent.py`
- `modeling/grpo_sharpe/network.py`
- `modeling/grpo_sharpe/enviroment.py`
- `modeling/grpo_sharpe/grpo.py`
- `modeling/grpo_sharpe/grpo_monthly.py`
- `modeling/grpo_sharpe/config/train_config.yaml`
- `modeling/grpo_sharpe/eval_monthly.py`
- `modeling/grpo_sharpe/eval_monthly_all_seed.ipynb`
- `modeling/grpo_sharpe/eval_monthly_remove_asset.ipynb`

## Benchmark Files

PPO benchmark:

- `modeling/ppo_benchmark/agent.py`
- `modeling/ppo_benchmark/network.py`
- `modeling/ppo_benchmark/run.py`
- `modeling/ppo_benchmark/run_monthly.py`
- `modeling/ppo_benchmark/eval_monthly.ipynb`
- `modeling/ppo_benchmark/eval_monthly_total.ipynb`
- `modeling/ppo_benchmark/eval_monthly_del_asset.ipynb`

SAC benchmark:

- `modeling/sac_benchmark/agent.py`
- `modeling/sac_benchmark/network.py`
- `modeling/sac_benchmark/buffer.py`
- `modeling/sac_benchmark/run.py`
- `modeling/sac_benchmark/run_monthly.py`
- `modeling/sac_benchmark/eval_monthly.ipynb`
- `modeling/sac_benchmark/eval_monthly_all.ipynb`
- `modeling/sac_benchmark/eval_monthly_del.ipynb`

## Experiment Notes

- The final sweep scripts use the Sharpe reward setting.
- Seed result files such as `dominant_one_hot_seed_test_*.xlsx` store strategy
  selection outputs with columns such as `risk_parity`, `min_var`,
  `max_sharpe`, `paa`, and `dominant_strategy`.
- Remove-asset experiments are marked with names such as `del_asset`, `remove`,
  `benchmark_del`, `grpo_del`, and
  `concat_portfolio_test_monthly_v2_remove.pt`.
