# Variational Imbalanced Regression on AgeDB-DIR (Review Paper Adaptation)

This directory contains an implementation of **Variational Imbalanced Regression (VIR)** on the *AgeDB-DIR* dataset, adapted for the review paper "Deconstructing Deep Imbalanced Regression".

*Original Repository:* [Wang et al., CVPR 2023](https://github.com/RxWang/VIR)

## Modifications from Original Repository

The following changes were made to the original code to unify the implementation and ensure reproducibility for the review paper:

### 1. Reproducibility & Seeding

- Added `seed_everything` function in `utils.py`.
- Added `worker_init_fn` in `utils.py`.
- Updated `rerun.py` to accept a `--seed` argument.
- Passed `worker_init_fn` and `generator` to `DataLoader` in `rerun.py`.

### 2. Logging & Results

- Added functionality in `rerun.py` (inside `validate`) to append evaluation results to a CSV file (`outputs.csv`).
- The CSV log includes: Run Name, MAE (All, Many, Median, Few), and G-Mean (All, Many, Median, Few).
- Added `save_age_bin_metrics_to_csv` in `utils.py`.
- Added age bin metrics calculation and CSV saving functionality.

### 3. Dataset & Protocols

- Updated `rerun.py` to load dataset splits from a CSV file specified by `--dataset` and `--data_dir`.
- Adjusted `datasets.py` to handle the new CSV format.
- Added support for `agedb_blindspot` argument.
- **Bug Fix**: Enabled VIR to run in blindspot regimes.

### 4. Hardware Support

- Added `torch.mps.manual_seed` in `seed_everything` in `utils.py`.
- Note: `rerun.py` primarily targets CUDA via `args.gpu`, but seeding support for MPS is present in utils.

### 5. Unification

- Standardized `shot_metrics` function in `utils.py`.
- Added error handling for shot metrics calculations.

---

## Original README Content

# Variational Imbalanced Regression

This is a preview version. Stay tuned for the official version.

The pretrained model and training log is available at [here](https://drive.google.com/drive/folders/1bbdzbvXJVgkIGe1AbsH9VEm0Uvkqp67R?usp=sharing).

```
2023-05-20 17:15:50,626 | Args: Namespace(
    augment=True, 
    batch_size=64, 
    best_loss=100000.0,
    bucket_num=100,
    bucket_start=3,
    data_dir='./data/', # Adjusted
    dataset='agedb', 
    epoch=90, 
    evaluate=False, 
    fds=True, 
    fds_kernel='gaussian', 
    fds_ks=5, 
    fds_mmt=0.9, 
    fds_sigma=2, 
    gpu=6, 
    img_size=224, 
    lambda_recons=0.7, 
    lambda_reg=0.01, 
    lds=True, 
    lds_kernel='gaussian', 
    lds_ks=5, 
    lds_sigma=2, 
    loss='l1', 
    lr=0.001, 
    model='resnet50', 
    momentum=0.9, 
    optimizer='adam', 
    pretrained='', 
    print_freq=10, 
    resume='', 
    retrain_fc=False, 
    reweight='sqrt_inv', 
    schedule=[60, 80], 
    seeds=1223, 
    start_epoch=0, 
    start_smooth=1, 
    start_update=0, 
    store_name='0.01_1223_0.7_agedb_resnet50_lds_gau_5_2_prm_gau_5_2_0_1_0.9_adam_l1_0.001_64_cdm_recons', 
    store_root='/data/local/ziyan/checkpoint', 
    use_cdm=True, 
    use_edl=True, 
    use_prm=True, 
    use_recons=True, 
    weight_decay=0.0001, 
    workers=32)
```
