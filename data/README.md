# Data Artifacts

Large datasets, tensor files, checkpoints, and W&B artifacts are not tracked in
Git. Place local artifacts under this directory before running training or
evaluation.

## Required CSV Files

Default train configs expect:

```text
data/before_train_v3.csv
data/train_v3.csv
data/before_future_train_v3.csv
data/future_train_v3.csv
```

Default test configs expect:

```text
data/train_v3.csv
data/test_v3.csv
data/future_train_v3.csv
data/future_test_v3.csv
```

## Required Tensor Files

```text
data/portfolio_price/concat_portfolio_train_monthly_v1.pt
data/portfolio_price/concat_portfolio_valid_monthly_v1.pt
data/portfolio_price/concat_portfolio_test_monthly_v1.pt
```

## Optional Robustness Artifacts

Asset-exclusion experiments used separate local files, for example:

```text
data/train_del_asset_v1.csv
data/test_del_asset_v1.csv
data/future_train_del_asset_v1.csv
data/future_test_del_asset_v1.csv
data/portfolio_price/concat_portfolio_test_monthly_v2_remove.pt
```

These are not required for the default training scripts.

## Checkpoints

Trained checkpoints are expected under `outputs/` by default:

```text
outputs/grpo_sharpe/final_grpo_model.pth
outputs/ppo_benchmark/final_ppo_model.pth
outputs/sac_benchmark/final_sac_model.pth
```

Update each `config/test_config.yaml` if local checkpoint names differ.
