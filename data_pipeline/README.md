# Data Pipeline

The data pipeline keeps only the final notebooks required for the paper's model
input surface.

| Step | Notebook | Output role |
|---|---|---|
| 1 | `01_prepare_price_universe.ipynb` | Prepare the 30-asset price universe from raw price inputs. |
| 2 | `02_create_chronological_splits.ipynb` | Create chronological train, validation, and test inputs. |
| 3 | `03_build_rl_state_tensor.ipynb` | Apply train-only min-max scaling and create the 52-dimensional RL state tensors. |

Exploratory analysis notebooks, earlier variant notebooks, robustness-only
asset-removal notebooks, and rendered notebook outputs are excluded from the
submitted code surface.
