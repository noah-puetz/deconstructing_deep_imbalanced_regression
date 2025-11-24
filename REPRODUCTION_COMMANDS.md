# Reproduction Commands

This document lists the Python commands required to reproduce the experiments.

### Vanilla Model

To train the Vanilla model:

```bash
python3 agedb_dir_conr/train.py \
    --epoch 120 \
    --batch_size 256 \
    --store_name backbone \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla[agedb]
```

### LDS Model

To train the model with Label Distribution Smoothing (LDS):

```bash
python3 agedb_dir_conr/train.py \
    --lds True \
    --lds_ks 5 \
    --lds_sigma 2 \
    --reweight sqrt_inv \
    --epoch 120 \
    --batch_size 256 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+lds[agedb]
```

### FDS Model

To train the model with Feature Distribution Smoothing (FDS):

```bash
python3 agedb_dir_conr/train.py \
    --fds \
    --epoch 120 \
    --batch_size 256 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+fds[agedb]
```

### LDS + FDS Model

To train the model with both LDS and FDS:

```bash
python3 agedb_dir_conr/train.py \
    --lds True \
    --lds_ks 5 \
    --lds_sigma 2 \
    --reweight sqrt_inv \
    --fds True \
    --epoch 120 \
    --batch_size 256 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+lds+fds[agedb]
```

### Focal L1 Loss Model

To train the model using Focal L1 Loss:

```bash
python3 agedb_dir_conr/train.py \
    --loss focal_l1 \
    --lr 2.5e-4 \
    --epoch 120 \
    --batch_size 64 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+focall1[agedb]
```

### SQINV Model

To train the model with Inverse Square Root reweighting:

```bash
python3 agedb_dir_conr/train.py \
    --reweight sqrt_inv \
    --epoch 120 \
    --batch_size 256 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+sqinv[agedb]
```

### RRT Model

To train the RRT model (requires a pretrained model, typically 'latest'):

```bash
python3 agedb_dir_conr/train.py --name rrt \
    --retrain_fc \
    --reweight sqrt_inv \
    --pretrained latest \
    --lr 0.001 \
    --epoch 30 \
    --batch_size 256 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+rrt[agedb]
```

### RRT with BNI

To train RRT with Batch Normalization Improvement (BNI):

```bash
python3 agedb_dir_conr/train.py --retrain_fc \
    --balanced_metric \
    --bmse \
    --imp bni \
    --init_noise_sigma 10. \
    --sigma_lr 0.001 \
    --lr 0.0001 \
    --epoch 10 \
    --batch_size 256 \
    --pretrained latest \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+rrt+bni[agedb]
```

### RRT with BMC

To train RRT with Balanced MSE (BMC):

```bash
python3 agedb_dir_conr/train.py --retrain_fc \
    --balanced_metric \
    --bmse \
    --imp bmc \
    --init_noise_sigma 10. \
    --sigma_lr 0.01 \
    --lr 0.0001 \
    --epoch 10 \
    --batch_size 256 \
    --pretrained latest \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+rrt+bmc[agedb]
```

### Vanilla + RankSim

To train Vanilla with RankSim regularization:

```bash
python3 agedb_dir_conr/train.py \
    --epoch 120 \
    --lr 2.5e-4 \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+ranksim[agedb]
```

### LDS + RankSim

To train LDS with RankSim:

```bash
python3 agedb_dir_conr/train.py \
    --lds True \
    --lds_ks 5 \
    --lds_sigma 2 \
    --reweight sqrt_inv \
    --lr 2.5e-4 \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --epoch 120 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+lds+ranksim[agedb]
```

### FDS + RankSim

To train FDS with RankSim:

```bash
python3 agedb_dir_conr/train.py \
    --fds \
    --lr 2.5e-4 \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --epoch 120 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+fds+ranksim[agedb]
```

### LDS + FDS + RankSim

To train LDS + FDS with RankSim:

```bash
python3 agedb_dir_conr/train.py \
    --lds True \
    --lds_ks 5 \
    --lds_sigma 2 \
    --reweight sqrt_inv \
    --fds True \
    --lr 2.5e-4 \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --epoch 120 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+lds+fds+ranksim[agedb]
```

### Focal L1 + RankSim

To train Focal L1 with RankSim:

```bash
python3 agedb_dir_conr/train.py \
    --epoch 120 \
    --loss focal_l1 \
    --lr 2.5e-4 \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+focall1+ranksim[agedb]
```

### SQINV + RankSim

To train SQINV with RankSim:

```bash
python3 agedb_dir_conr/train.py \
    --epoch 120 \
    --lr 2.5e-4 \
    --reweight sqrt_inv \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+sqinv+ranksim[agedb]
```

### RankSim Backbone (Pre-training)

To train the RankSim backbone (needed for RRT+RankSim):

```bash
python3 agedb_dir_conr/train.py \
    --epoch 120 \
    --lr 2.5e-4 \
    --regularization_weight=100.0 \
    --interpolation_lambda=2.0 \
    --store_name backbone \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name ranksim_backbone
```

### Vanilla + RRT + RankSim

To train RRT with RankSim (uses pretrained backbone):

```bash
python3 agedb_dir_conr/train.py \
    --retrain_fc \
    --reweight sqrt_inv \
    --pretrained latest \
    --epoch 30 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+rrt+ranksim[agedb]
```

### Vanilla + ConR

To train Vanilla with ConR:

```bash
python3 agedb_dir_conr/train.py \
    --conr True \
    --lr 2.5e-4 \
    --epoch 120 \
    --batch_size 64 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+conr[agedb]
```

### LDS + ConR

To train LDS with ConR:

```bash
python3 agedb_dir_conr/train.py \
    --conr True \
    --lr 2.5e-4 \
    --lds True \
    --lds_ks 5 \
    --lds_sigma 2 \
    --reweight sqrt_inv \
    --epoch 120 \
    --batch_size 64 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+lds+conr[agedb]
```

### FDS + ConR

To train FDS with ConR:

```bash
python3 agedb_dir_conr/train.py \
    --conr True \
    --lr 2.5e-4 \
    --fds \
    --epoch 120 \
    --batch_size 64 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+fds+conr[agedb]
```

### LDS + FDS + ConR

To train LDS + FDS with ConR:

```bash
python3 agedb_dir_conr/train.py \
    --conr True \
    --lr 2.5e-4 \
    --lds True \
    --lds_ks 5 \
    --lds_sigma 2 \
    --reweight sqrt_inv \
    --fds True \
    --epoch 120 \
    --batch_size 64 \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vanilla+lds+fds+conr[agedb]
```

### UVOTE Model

To train the UVOTE model:

```bash
python3 agedb_dir_uvote/train_gradual.py \
    --loss l1nll \
    --dynamic_loss \
    --name UVOTE \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name uvote[agedb]
```

### VIR Model

To train the VIR model:

```bash
python3 agedb_dir_vir/rerun.py \
    --name VIR \
    --dataset agedb \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name vir[agedb]
```

### RnC Model (Training)

To train the RnC model:

```bash
python3 agedb_dir_rnc/main_rnc.py \
    --epoch 400 \
    --model resnet50 \
    --split_csv data/agedb.csv \
    --seed 42
```

### RnC Model (Linear Evaluation)

To evaluate the RnC model (Linear Probing):

```bash
python3 agedb_dir_rnc/main_linear.py \
    --model resnet50 \
    --ckpt latest \
    --split_csv data/agedb.csv \
    --output_csv "agedb_output_seed_42.csv" \
    --seed 42 \
    --name RnC[agedb]
```
