# UVOTE on AgeDB-DIR (Review Paper Adaptation)

This directory contains an implementation of **UVOTE** (Uncertainty-aware Voting) on the *AgeDB-DIR* dataset, adapted for the review paper "Deconstructing Deep Imbalanced Regression".

*Original Repository:* [Jiang et al., CVPR 2023](https://github.com/Yuj1ang/MOUV)

## Modifications from Original Repository

The following changes were made to the original code to unify the implementation and ensure reproducibility for the review paper:

### 1. Reproducibility & Seeding

- Added `seed_everything` function in `utils.py`.
- Added `seed_worker` function in `utils.py`.
- Updated `train_gradual.py` to accept a `--seed` argument.
- Passed `worker_init_fn` and `generator` to `DataLoader` in `train_gradual.py`.

### 2. Logging & Results

- Added functionality in `train_gradual.py` (inside `validate_mul`) to append evaluation results to a CSV file (`outputs.csv`).
- The CSV log includes: Run Name, MAE (All, Many, Median, Few), and G-Mean (All, Many, Median, Few).
- Integrated `wandb` logging (optional).

### 3. Dataset & Protocols

- Updated `train_gradual.py` to load dataset splits from a CSV file specified by `--dataset` and `--data_dir`.
- Adjusted `datasets_gradual.py` to handle the new CSV format.
- Added support for `agedb_blindspot` argument.
- **Bug Fix**: Enabled UVOTE to run in blindspot regimes.

### 4. Hardware Support

- Added support for Apple Silicon (MPS) devices in `train_gradual.py`.
- Added `torch.mps.manual_seed` in `seed_everything`.

### 5. Unification

- Standardized `shot_metrics` function in `utils.py`.
- Added error handling for shot metrics calculations.

---

## Original README Content

# AgeDB-DIR

## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data`

2. **(Optional)** We have provided required AgeDB-DIR meta file `agedb.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. If you want to try other different balanced splits, you can generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL

## Code Overview

#### Main Files

- `train_gradual.py`: main training and evaluation script
- `create_agedb.py`: create AgeDB raw meta data
- `preprocess_agedb.py`: create AgeDB-DIR meta file `agedb.csv` with balanced val/test set

## Main arguments

- `--data_dir`: data directory to place data and meta file
- `--num_branch`: number of branch for model
- `--loss`: training loss type
- `--resume`: path to resume checkpoint (for both training and evaluation)
- `--evaluate`: evaluate only flag

#### Training

```bash
# for example, train with 2-expert model
python train_gradual.py --loss l1nll --num_branch 2 --dynamic_loss
```

#### Evaluation

```bash
python train_gradual.py --evaluate --resume MODEL_CHECKPOINT  [other model settings: e.g.--loss l1nll --num_branch 2]
```

## Pretrained model

- [model for agedb](https://share.phys.ethz.ch/~pf/yujiangdata/mouv/agedb-dir/agedb_resnet50_2_dyL_adam_l1nll.zip)
