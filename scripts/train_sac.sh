#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/backtest:${PYTHONPATH:-}"

cd "${ROOT_DIR}/modeling/sac_benchmark"
python train_5d_discrete_rebal_step_seed.py
