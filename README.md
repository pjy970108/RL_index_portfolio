# RL Index Portfolio

Official code for **Strategy-Level Reinforcement Learning Framework for
Portfolio Management**.

The final model is a GRPO-based portfolio strategy selector. It uses
Sharpe-based environment rewards and min-max normalized group returns for GRPO
advantage estimation. PPO and SAC are included as benchmark methods.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)

## Overview

The project trains reinforcement-learning agents to select among portfolio
strategies over monthly rebalancing periods.

Main workflow:

1. Build portfolio price and feature tensors from local market data.
2. Train the final GRPO model with Sharpe reward.
3. Train PPO and SAC benchmark models.
4. Evaluate trained checkpoints on the test period.
5. Use the retained paper figures for thesis reporting.

The GRPO update converts trajectory-level group returns into min-max normalized
advantages:

```text
advantage = (group_return - min(group_returns)) / (max(group_returns) - min(group_returns))
```

This is the min-max component used by the final GRPO training code. It is
separate from feature min-max normalization in the data pipeline.

## Project Structure

```text
RL_index_portfolio/
|
|-- data_pipeline/                 # Local data construction notebooks
|-- modeling/
|   |-- grpo_sharpe/               # Final GRPO implementation
|   |-- ppo_benchmark/             # PPO benchmark
|   `-- sac_benchmark/             # SAC benchmark
|-- backtest/                      # Strategy return and metric utilities
|-- result/figures/paper_figures/  # Minimal paper figure assets
|-- scripts/                       # Shell entry points
|-- data/README.md                 # Local data contract
|-- CODE_MAP.md
`-- requirements.txt
```

## Requirements

- Python 3.10 or later is recommended.
- Install a PyTorch build that matches the local CUDA or CPU environment.
- W&B is used for training sweeps. Use `wandb login` for online tracking, or
  set `WANDB_MODE=offline` for local/offline runs.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone https://github.com/pjy970108/RL_index_portfolio.git
cd RL_index_portfolio
conda create -n rl-index-portfolio python=3.10
conda activate rl-index-portfolio
pip install -r requirements.txt
```

## Data

Large datasets, feature tensors, trained checkpoints, and W&B run directories
are not tracked in Git.

Place local data under `data/` according to [data/README.md](data/README.md).
The default configs expect files such as:

- `data/train_v3.csv`
- `data/test_v3.csv`
- `data/future_train_v3.csv`
- `data/future_test_v3.csv`
- `data/portfolio_price/concat_portfolio_train_monthly_v1.pt`
- `data/portfolio_price/concat_portfolio_valid_monthly_v1.pt`
- `data/portfolio_price/concat_portfolio_test_monthly_v1.pt`

Trained checkpoints are expected under `outputs/` by default. The directory is
ignored by Git.

## Usage

Run commands from the repository root.

### Train GRPO

```bash
bash scripts/train_grpo.sh
```

Main files:

- `modeling/grpo_sharpe/train_concat_monthly_1m_discrete_rebal_step_batch_sample_seed_change.py`
- `modeling/grpo_sharpe/grpo_monthly.py`
- `modeling/grpo_sharpe/enviroment.py`

### Evaluate GRPO

```bash
bash scripts/eval_grpo.sh
```

Before evaluation, place the trained checkpoint at the path configured in:

```text
modeling/grpo_sharpe/config/test_config.yaml
```

### Train Benchmarks

```bash
bash scripts/train_ppo.sh
bash scripts/train_sac.sh
```

Benchmark files:

- `modeling/ppo_benchmark/train_5d_discrete_rebal_step_seed.py`
- `modeling/sac_benchmark/train_5d_discrete_rebal_step_seed.py`

## Results

Only minimal paper figure assets are retained in:

```text
result/figures/paper_figures/
```

Intermediate notebooks, W&B artifacts, model checkpoints, and large tabular
outputs are intentionally excluded from the submitted code surface.

## Citation

Citation information can be added after the paper metadata is finalized.
