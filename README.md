# RL Index Portfolio

## Overview

Official implementation of reinforcement-learning based index portfolio
strategy selection.

Pipeline:

1. Data preprocessing — Build portfolio price, indicator, and train/test tensor artifacts
2. GRPO training — Train the final Sharpe-based GRPO strategy selector
3. Benchmark training — Train PPO and SAC benchmark methods
4. Evaluation — Run monthly portfolio evaluation, seed checks, and remove-asset tests
5. Backtest analysis — Compute portfolio performance metrics and generate result figures

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Data Availability](#data-availability)
- [Citation](#citation)

---

## Project Structure

```text
RL_index_portfolio/
|
├── data_pipeline/                     # Portfolio data and tensor construction
│   ├── make_indicator.ipynb
│   ├── make_indicator_no_tarn.ipynb
│   ├── make_portfolio_pricing_total_price.ipynb
│   ├── make_portfolio_pricing_total_price_v2_asset_del.ipynb
│   └── eda_distribution/
│
├── modeling/                          # Model training and evaluation
│   ├── grpo_sharpe/                   # Main GRPO Sharpe implementation
│   │   ├── agent.py
│   │   ├── network.py
│   │   ├── enviroment.py
│   │   ├── grpo.py
│   │   ├── grpo_monthly.py
│   │   ├── eval_monthly.py
│   │   └── config/
│   ├── ppo_benchmark/                 # PPO benchmark
│   └── sac_benchmark/                 # SAC benchmark
│
├── backtest/                          # Portfolio strategies and metrics
│   ├── dynamic_portfolio.py
│   ├── dynamic_portfolio_monthly.py
│   ├── backtesting_all_asset.py
│   ├── backtesting_all_asset_monthly.py
│   └── eval_metric.py
│
├── figure/                            # Figure and result analysis notebooks
│   ├── total_plot.ipynb
│   ├── total_plot_total_seed.ipynb
│   ├── total_average_remove_asset_model.ipynb
│   └── total_sensitive_test.ipynb
│
├── result/                            # Generated result figures and analysis assets
│   └── figures/
│
├── scripts/                           # Shell entry points
│   ├── train_grpo.sh
│   ├── eval_grpo.sh
│   ├── train_ppo.sh
│   └── train_sac.sh
│
├── data/                              # Local data placeholder; large files are not tracked
│   └── README.md
│
├── requirements.txt
├── CODE_MAP.md
└── README.md
```

---

## Requirements

- Python 3.10 or later is recommended.
- PyTorch should match the local CUDA or CPU environment.

Main dependencies:

- `torch`
- `pandas`, `numpy`, `scipy`
- `PyYAML`, `joblib`, `tqdm`
- `cvxpy`
- `statsmodels`, `arch`
- `wandb` for experiment tracking
- `openpyxl` for Excel result files

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/pjy970108/RL_index_portfolio.git
cd RL_index_portfolio
```

### 2. Create a Python environment

```bash
conda create -n rl-index-portfolio python=3.10
conda activate rl-index-portfolio
```

### 3. Install PyTorch

Install PyTorch for the local CUDA or CPU environment.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

All shell scripts should be run from the project root. The scripts set
`PYTHONPATH` so that model code can import the shared files in `backtest/`.

### Step 1 — Data Preprocessing

Run the notebooks in `data_pipeline/` to construct local tensor files.

Important notebooks:

- `data_pipeline/make_indicator_no_tarn.ipynb`
- `data_pipeline/make_portfolio_pricing_total_price.ipynb`
- `data_pipeline/make_portfolio_pricing_total_price_v2_asset_del.ipynb`

### Step 2 — GRPO Training

```bash
bash scripts/train_grpo.sh
```

Main implementation:

```text
modeling/grpo_sharpe/
```

### Step 3 — PPO/SAC Benchmark Training

```bash
bash scripts/train_ppo.sh
bash scripts/train_sac.sh
```

Benchmark implementations:

```text
modeling/ppo_benchmark/
modeling/sac_benchmark/
```

### Step 4 — GRPO Evaluation

```bash
bash scripts/eval_grpo.sh
```

Additional evaluation notebooks:

- `modeling/grpo_sharpe/eval_monthly_all_seed.ipynb`
- `modeling/grpo_sharpe/eval_monthly_remove_asset.ipynb`
- `modeling/grpo_sharpe/eval_sub_period.ipynb`

### Step 5 — Result Figures

Run the notebooks in `figure/` to generate final plots and summary figures.

Generated figure assets are stored under:

```text
result/figures/
```

---

## Data Availability

Large datasets, feature tensors, trained checkpoints, and W&B run directories
are not included in this repository.

Expected local artifacts are described in:

```text
data/README.md
```

The ignored local artifacts include:

- `data/*`
- `*.csv`
- `*.pt`
- `*.pth`
- `wandb/`

---

## Notes

- GRPO is the final model path.
- PPO and SAC are benchmark methods.
- The final GRPO sweeps use the Sharpe reward setting.
- `train_concat_monthly_1m_discrete_rebal_step_batch_sample_seed_change4.py`
  is a seed/small-run script, not the main full training entry point.

---

## Citation

Citation information can be added after the paper metadata is finalized.
