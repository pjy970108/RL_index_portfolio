#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/backtest:${PYTHONPATH:-}"

cd "${ROOT_DIR}/modeling/grpo_sharpe"
python eval_monthly.py
