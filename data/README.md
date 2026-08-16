# Data Artifacts

Large datasets, feature tensors, model checkpoints, and W&B artifacts are kept
outside Git.

The research scripts reference local artifacts such as:

- portfolio price and feature CSV files
- train/test tensor files with `.pt` extensions
- remove-asset tensor files such as `concat_portfolio_test_monthly_v2_remove.pt`
- trained model checkpoints with `.pth` extensions
- W&B run directories

Expected ignored paths and extensions:

```text
data/
*.csv
*.pt
*.pth
wandb/
```

Before running training or evaluation scripts, place the required local data and
checkpoint files at the paths referenced by the corresponding script or update
the script/config paths for your local environment.
