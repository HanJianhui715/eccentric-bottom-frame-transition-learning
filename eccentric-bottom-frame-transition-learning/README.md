# Monotonic state-transition learning dataset and code

This repository is a minimal review package for the manuscript on low-mid-high performance-state transition prediction of eccentric bottom-frame structures.

Only the essential code and three data tables are included. Generated figures, model-output tables and draft notes are intentionally excluded.

## Data tables

All tables are in `data/`.

| File | Rows | Purpose |
|---|---:|---|
| `raw_fe_records_339.csv` | 339 | Original finite-element point-response records with column headers. |
| `state_sequences_113.csv` | 113 | Concise structure-level L-M-H state-sequence table for inspection. |
| `ml_transition_samples_226.csv` | 226 | Direct machine-learning table for one-step transition learning and rollout. |

`raw_fe_records_339.csv` columns:

```text
SPI, ISR, Ecc_X_1, Ecc_Y_1, Ecc_X_2, Ecc_Y_2, PGA, IDR_frame, IDR_masonry
```

The manuscript uses the following notation for the seven input variables:

```text
SPI = seismic precautionary intensity
ISR = inter-story stiffness ratio
EX1 = first-story eccentricity in the X direction = Ecc_X_1 in the CSV files
EY1 = first-story eccentricity in the Y direction = Ecc_Y_1 in the CSV files
EX2 = second-story eccentricity in the X direction = Ecc_X_2 in the CSV files
EY2 = second-story eccentricity in the Y direction = Ecc_Y_2 in the CSV files
PGA = peak ground acceleration
```

`ml_transition_samples_226.csv` contains the implemented model input features:

```text
SPI, ISR, Ecc_X_1, Ecc_Y_1, Ecc_X_2, Ecc_Y_2,
from_pga, to_pga, pga_ratio, log_pga_ratio,
from_log_IDR_frame, from_log_IDR_masonry,
from_damage_frame, from_damage_masonry, from_damage_global,
transition_stage
```

The target columns are:

```text
target_log_delta_frame
target_log_delta_masonry
```

where the implemented target transform is:

```text
target_log_delta = log(max(log(theta_j / theta_i), EPS))
```

After prediction, the target-stage IDR is updated as:

```text
theta_hat_j = theta_i * exp(exp(predicted_target))
```

`transition_stage` is encoded as `0` for L->M and `1` for M->H.

## Code

The essential scripts are in `scripts/`.

| Script | Purpose |
|---|---|
| `01_build_transition_dataset.py` | Rebuilds `state_sequences_113.csv` and `ml_transition_samples_226.csv` from `raw_fe_records_339.csv`. |
| `02_run_transition_emulator.py` | Runs the main one-step transition models and two-step rollout. |
| `08_run_feature_ablation.py` | Runs the feature-set ablation experiment. |
| `transition_common.py` | Shared thresholds, metrics, split utilities and model factories. |

The damage-threshold definitions and raw-data loader are in `src/damage_assessment/data.py`.

## Reproduction

Install dependencies:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Rebuild the two derived data tables:

```bash
.venv/Scripts/python scripts/01_build_transition_dataset.py
```

Run a dry check of the main experiment setup:

```bash
.venv/Scripts/python scripts/02_run_transition_emulator.py --dry-run --device cpu
```

Run the main experiment:

```bash
.venv/Scripts/python scripts/02_run_transition_emulator.py --models tabpfn,rf,lightgbm,xgboost --protocols groupkfold,leave_spi --n-splits 5 --members 3 --tabpfn-estimators 1 --device cpu --random-state 42 --rf-trees 500 --rf-min-leaf 1 --gbdt-trees 450 --gbdt-learning-rate 0.03 --predict-batch-size 8 --with-probabilities
```

Run the feature-set ablation:

```bash
.venv/Scripts/python scripts/08_run_feature_ablation.py --models tabpfn,rf,lightgbm,xgboost --protocols groupkfold,leave_spi --n-splits 5 --members 1 --tabpfn-estimators 1 --device cpu --random-state 42 --rf-trees 500 --rf-min-leaf 1 --gbdt-trees 450 --gbdt-learning-rate 0.03 --predict-batch-size 8
```

Model outputs are written to `results/`, which is ignored by git.

## Validation split

The main validation scheme is five-fold GroupKFold using `group_id` as the group identifier. Thus, the L->M and M->H transition samples from the same structural group are always assigned to the same fold.

Leave-SPI-out is retained only as a supplementary stress-test protocol.
