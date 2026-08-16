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
- [Data Pipeline](#data-pipeline)
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
|-- data_pipeline/                 # Final data construction notebooks
|-- modeling/
|   |-- grpo_sharpe/               # Final GRPO implementation
|   |-- ppo_benchmark/             # PPO benchmark
|   `-- sac_benchmark/             # SAC benchmark
|-- backtest/                      # Strategy return and metric utilities
|-- result/figures/paper_figures/  # Minimal paper figure assets
|-- scripts/                       # Shell entry points
|-- DATA_CONTRACT.md               # Data period and feature contract
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

The paper uses price data from February 2002 through December 2024. Because
strategy returns and state features require a 252-trading-day lookback, the
effective model sample starts in February 2003.

| Split | Period | Role |
|---|---|---|
| Raw data | 2002.02-2024.12 | Full source price history |
| Train | 2003.02.19-2013.12.31 | RL training |
| Validation | 2014.01.01-2018.12.31 | Early stopping and model selection |
| Test | 2019.01.01-2024.12.31 | Final out-of-sample evaluation |

See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the data and feature contract.

## Data Pipeline

The submitted data pipeline keeps only the final notebooks needed to reproduce
the model input surface:

1. `data_pipeline/01_prepare_price_universe.ipynb`
2. `data_pipeline/02_create_chronological_splits.ipynb`
3. `data_pipeline/03_build_rl_state_tensor.ipynb`

The pipeline prepares the 30-asset price universe, creates chronological
train/validation/test inputs, builds four strategy portfolios with a
252-trading-day lookback, derives 13 portfolio-level features per strategy, and
applies train-set-only min-max scaling to form the 52-dimensional RL state.

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
