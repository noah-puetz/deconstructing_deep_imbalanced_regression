# ConR on AgeDB-DIR (Review Paper Adaptation)

This directory contains an implementation of **ConR** on the *AgeDB-DIR* dataset, adapted for the review paper "Deconstructing Deep Imbalanced Regression".

The original code is based on the implementation of [ConR](https://github.com/BorealisAI/ranksim-imbalanced-regression).

## Modifications from Original Repository

The following changes were made to the original code to unify the implementation and ensure reproducibility for the review paper:

### 1. Reproducibility & Seeding

- Added `seed_everything` function in `utils.py` to set seeds for `random`, `numpy`, `torch`, and `torch.cuda`.
- Added `seed_worker` function in `utils.py` for DataLoader worker seeding.
- Updated `train.py` to accept a `--seed` argument (default: 42).
- Initialized `torch.Generator` with the seed and passed it to `DataLoader` to ensure deterministic data loading.

### 2. Logging & Results

- Added functionality in `train.py` to append evaluation results to a CSV file (`outputs.csv`).
- The CSV log includes: Run Name, MAE (All, Many, Median, Few), and G-Mean (All, Many, Median, Few).
- Added `save_age_bin_metrics_to_csv` in `utils.py` for detailed age-bin metrics.
- Added logging of model parameter size.

### 3. Dataset & Protocols

- Updated `train.py` to load dataset splits from a CSV file specified by `--dataset` and `--data_dir`.
- Added support for new dataset variants (e.g., `agedb_1_gaussian`, `agedb_blindspot`).
- Adjusted `AgeDB` dataset class in `datasets.py` to handle the new CSV format and directory structure.
- Added Balanced MSE (BMSE) loss implementation and related arguments.

### 4. Hardware Support

- Added support for Apple Silicon (MPS) devices in `train.py`.
- Automatically detects `cuda`, `mps`, or `cpu`.
- Refactored device selection logic.

### 5. Unification

- Standardized `shot_metrics` function in `utils.py` to be consistent with other methods in the review.
- Added error handling for shot metrics calculations.

---

## Original README Content

# ConR on AgeDB-DIR

This repository contains the implementation of **ConR** on *AgeDB-DIR* dataset.

The imbalanced regression framework and LDS+FDS are based on the public repository of [Gong et al., ICML 2022](https://github.com/BorealisAI/ranksim-imbalanced-regression).

## Installation

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data`

2. We use the standard train/val/test split file (`agedb.csv` in folder `./data`) provided by Yang et al.(ICML 2021), which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Dependencies

- PyTorch (>= 1.2, tested on 1.6)
- tensorboard_logger
- numpy, pandas, scipy, tqdm, matplotlib, PIL, wget

## Code Overview

#### Main Files

- `train.py`: main training and evaluation script
- `create_agedb.py`: create AgeDB raw meta data
- `preprocess_agedb.py`: create AgeDB-DIR meta file `agedb.csv` with balanced val/test set

#### Main Arguments

- `--data_dir`: data directory to place data and meta file
- `--reweight`: cost-sensitive re-weighting scheme to use
- `--loss`: training loss type
- `--conr`: wether to use ConR or not.
- `-w`: distance threshold (default 1.0)
- `--beta`: the scale of ConR loss (default 4.0)
- `-t`: temperature(default 0.2)
- `-e`: pushing power scale(default 0.01)

## Getting Started

### 1. Train baselines

To use Vanilla model

```bash
python train.py --batch_size 64 --lr 2.5e-4
```

### 2. Train a model with ConR

##### batch size 64, learning rate 2.5e-4

```bash
python train.py --batch_size 64 --lr 2.5e-4 --conr -w 1.0 --beta 4.0 -e 0.01
```

### 3. Evaluate and reproduce

If you do not train the model, you can evaluate the model and reproduce our results directly using the pretrained weights from the anonymous links below.

```bash
python train.py --evaluate [...evaluation model arguments...] --resume <path_to_evaluation_ckpt>
```
