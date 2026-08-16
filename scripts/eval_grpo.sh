#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/backtest:${PYTHONPATH:-}"

python "${ROOT_DIR}/modeling/grpo_sharpe/eval_monthly.py"
