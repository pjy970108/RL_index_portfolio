# RL Index Portfolio

This repository contains the research code for reinforcement-learning based
index portfolio strategy selection. It is organized around the final GRPO
Sharpe experiment, with PPO and SAC implementations kept as comparison
baselines.

Large local datasets, feature tensors, trained model checkpoints, and W&B run
artifacts are intentionally excluded from Git.

## Project Scope

The project studies reinforcement-learning based portfolio strategy selection.
The final research direction uses GRPO with a Sharpe-based reward, while PPO and
SAC are kept as comparison baselines.

At a high level, the agent selects among existing dynamic portfolio strategies:

- `risk_parity`
- `min_var`
- `max_sharpe`
- `paa`

The selected strategy returns are evaluated with portfolio performance metrics
such as annual return, Sharpe, Sortino, Calmar, drawdown, and volatility.

## Repository Layout

```text
core/
  portfolio_strategies/        Shared backtesting, dynamic portfolio, and metric code

experiments/
  grpo_sharpe/                 Final GRPO Sharpe-centered experiment

baselines/
  ppo/                         PPO comparison baseline experiments
  sac/                         SAC comparison baseline experiments

notebooks/
  data_preprocessing/          Data construction and feature tensor notebooks
  eda_distribution/            EDA and distribution checks
  final_figures/               Final reporting / paper figure notebooks

results/
  figures/                     Generated figures and image assets

legacy/
  grpo_concat_asset_original/  Earlier concat-asset GRPO experiment folder
  old_grpo/                    Earlier GRPO prototype code
  delete_candidates/           Empty or obsolete files kept for review
```

## Main Code Path

The final GRPO experiment is located under:

```text
experiments/grpo_sharpe/
```

Important files:

- `agent.py`: PPO-style policy wrapper used by GRPO training/evaluation code.
- `network.py`: actor network for discrete strategy selection.
- `enviroment.py`: portfolio environment using dynamic strategy backtests.
- `grpo.py`, `grpo_monthly.py`: GRPO rollout, update, and evaluation routines.
- `config/train_config.yaml`: shared training configuration baseline.
- `eval_monthly.py`: monthly evaluation script with the final Sharpe model path used in later experiments.

Note: the folder name `enviroment.py` is preserved from the original research
code to avoid rewriting history.

## Baselines

PPO and SAC are retained as comparison baselines:

- `baselines/ppo/`
- `baselines/sac/`

These folders contain multiple historical variants and comparison context.

## Additional Experiments

The GRPO Sharpe folder also includes:

- seed sweep and seed analysis notebooks, including `eval_monthly_all_seed.ipynb`
- remove-asset evaluation notebooks, including `eval_monthly_remove_asset.ipynb`
- `result/grpo_del/` and `result/benchmark_del/` for remove-asset result analysis
- strategy-selection outputs such as `dominant_one_hot_seed_test_*.xlsx`

## Data and Model Files

The following are intentionally not stored in Git:

- `data/`
- `*.csv`
- `*.pt`
- `*.pth`
- `wandb/`

The code and notebooks may reference local paths for these artifacts. Those
paths document the research workflow, while the large local artifacts are
managed outside this Git repository.

## Historical or External Dependencies

Some legacy scripts reference modules that are not part of this repository,
including `TARN_MAPPO`, `grpo_min_max`, `utils`, and `tarn.TARN`. These belong
to earlier experiment variants and should not be treated as the final GRPO
Sharpe code path.

See `CODE_MAP.md` for the detailed file classification.
