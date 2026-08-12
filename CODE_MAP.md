# Code Map

This map explains how the research code is organized. The main path is the
final GRPO Sharpe experiment, with baseline models, preprocessing notebooks,
figures, and earlier experiment variants separated by role.

| Category | Location | Notes |
|---|---|---|
| Final GRPO Sharpe experiment | `experiments/grpo_sharpe/` | Main research code path. Uses GRPO with Sharpe reward in sweep scripts. |
| Shared portfolio/backtest code | `core/portfolio_strategies/` | Dynamic portfolio strategies, partial backtests, and performance metrics. |
| PPO baseline | `baselines/ppo/` | Comparison baseline variants. Includes original, v1, and Sharpe-centered folders. |
| SAC baseline | `baselines/sac/` | Comparison baseline variants. Includes original and Sharpe-centered folders. |
| Data preprocessing notebooks | `notebooks/data_preprocessing/` | Data cleaning, indicator, and portfolio tensor construction notebooks. |
| Remove-asset preprocessing | `notebooks/data_preprocessing/make_portfolio_pricing_total_price_v2_asset_del.ipynb` | Builds remove-asset data artifacts referenced by later evaluation notebooks. |
| Final figure notebooks | `notebooks/final_figures/` | Reporting and paper-style plot notebooks. |
| EDA notebooks | `notebooks/eda_distribution/` | Distribution and sampling checks. |
| Figure outputs | `results/figures/` | PNG assets and generated figure directory. |
| Earlier concat-asset GRPO | `legacy/grpo_concat_asset_original/` | Earlier or original concat-asset experiment folder. Core code matches the Sharpe folder, but result/evaluation artifacts differ. |
| Older GRPO prototype | `legacy/old_grpo/grpo/` | Earlier GRPO implementation with `tarn.TARN` dependency. |
| Delete candidates | `legacy/delete_candidates/` | Empty or obsolete files separated for review. |

## Final GRPO Notes

The following files are identical between the former `concat_asset` and
`concat_sharpe` folders:

- `agent.py`
- `network.py`
- `enviroment.py`
- `grpo.py`
- `grpo_monthly.py`
- `config/train_config.yaml`

The `concat_sharpe` folder was kept as the main experiment because it contains
the later Sharpe-centered evaluation notebooks, seed analysis artifacts, and
remove-asset result folders.

## Seed and Remove-Asset Results

Seed result files such as `dominant_one_hot_seed_test_*.xlsx` store strategy
selection outputs with columns like:

- `risk_parity`
- `min_var`
- `max_sharpe`
- `paa`
- `dominant_strategy`

Remove-asset experiments are identified by paths or filenames containing:

- `del_asset`
- `remove`
- `benchmark_del`
- `grpo_del`
- `concat_portfolio_test_monthly_v2_remove.pt`

## Code Notes

- `config/train_config.yaml` may show `reward_cond: "combined_reward"`, but the final sweep scripts override it with `reward_cond: "sharpe"`.
- Some training scripts still contain historical output paths pointing to the old folder names.
- `train_concat_monthly_1m_discrete_rebal_step_batch_sample_seed_change4.py` is kept as a seed-test or small-run script, not as the main full training run.
- Scripts importing `TARN_MAPPO`, `grpo_min_max`, `utils`, or `tarn.TARN` are historical or externally dependent and are not part of the final GRPO Sharpe path.
