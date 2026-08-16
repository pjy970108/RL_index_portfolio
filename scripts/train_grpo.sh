#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/backtest:${PYTHONPATH:-}"

cd "${ROOT_DIR}/modeling/grpo_sharpe"
python train_concat_monthly_1m_discrete_rebal_step_batch_sample_seed_change.py
