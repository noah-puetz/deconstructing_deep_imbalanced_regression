# Rank-N-Contrast on AgeDB-DIR (Review Paper Adaptation)

This directory contains an implementation of **Rank-N-Contrast (RnC)** on the *AgeDB-DIR* dataset, adapted for the review paper "Deconstructing Deep Imbalanced Regression".

*Original Repository:* [Zha et al., NeurIPS 2023](https://github.com/KaiwenZha/Rank-N-Contrast)

## Modifications from Original Repository

The following changes were made to the original code to unify the implementation and ensure reproducibility for the review paper:

### 1. Reproducibility & Seeding

- Added `seed_everything` function in `utils.py`.
- Added `seed_worker` function in `main_rnc.py` and `main_linear.py`.
- Updated `main_rnc.py` and `main_linear.py` to accept a `--seed` argument.
- Passed `worker_init_fn` and `generator` to `DataLoader` in `main_rnc.py` and `main_linear.py`.

### 2. Logging & Results

- Added functionality in `main_linear.py` to append evaluation results to a CSV file (`outputs.csv`).
- The CSV log includes: Run Name, MAE (All, Many, Median, Few), and G-Mean (All, Many, Median, Few).

### 3. Dataset & Protocols

- Updated `main_rnc.py`, `main_l1.py`, and `main_linear.py` to accept `--split_csv` and `--data_folder` arguments.
- Adjusted `dataset.py` to load data based on the provided CSV split file.
- **Bug Fix**: Corrected `main_rnc.py` to test on the test set instead of the validation set.
- Added feature normalization option (`--normalize`) in `main_rnc.py`.

### 4. Hardware Support

- Added support for Apple Silicon (MPS) devices in `main_rnc.py`, `main_l1.py`, and `main_linear.py`.
- Added `torch.mps.manual_seed` in `seed_everything`.
- Refactored device selection logic.

### 5. Unification

- Standardized `shot_metrics` logic (embedded in `final_validation` in `main_linear.py`) to be consistent with other methods.
- Added error handling for shot metrics calculations.

---

## Original README Content

## Rank-N-Contrast: Learning Continuous Representations for Regression

[Paper](https://arxiv.org/abs/2210.01189) | [Talk](https://youtu.be/T7TCBDmxMO0?si=l4SBKnIu26k9uqVH) | [Slides](assets/slides.pdf) | [BibTex](assets/bibtex.txt)

<img src='assets/teaser.png'>

Rank-N-Contrast: Learning Continuous Representations for Regression\
[Kaiwen Zha](https://people.csail.mit.edu/kzha/)\*, [Peng Cao](https://people.csail.mit.edu/pengcao/)\*, [Jeany Son](https://jeanyson.github.io/), [Yuzhe Yang](https://www.mit.edu/~yuzhe/), [Dina Katabi](http://people.csail.mit.edu/dina/) (*equal contribution)\
*NeurIPS 2023 (Spotlight)*

### Loss Function

The loss function [`RnCLoss`](./loss.py#L34) in [`loss.py`](./loss.py) takes `features` and `labels` as input, and return the loss value.

```python
from loss import RnCLoss

# define loss function with temperature, label difference measure, 
# and feature similarity measure
criterion = RnCLoss(temperature=2, label_diff='l1', feature_sim='l2')

# features: [bs, 2, feat_dim]
features = ...
# labels: [bs, label_dim]
labels = ...

# compute RnC loss
loss = criterion(features, labels)
```

### Running

Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data`.

- To train the model with the L1 loss, run

    ```
    python main_l1.py
    ```

- To train the model with the RnC framework, first run

    ```
    python main_rnc.py
    ```

    to train the encoder. The checkpoint of the encoder will be saved to `./save`. Then, run

    ```
    python main_linear.py --ckpt <PATH_TO_THE_TRAINED_ENCODER_CHECKPOINT>
    ```

  to train the regressor.

### Model Checkpoints

The checkpoints of the encoder and the regressor trained on AgeDB dataset are available [here](https://drive.google.com/file/d/11_W-wArbk5lgTCKyJY0fsPoALbx_Qkno/view?usp=sharing).

### Citation

If you use this code for your research, please cite our paper:

```bibtex
@inproceedings{zha2023rank,
    title={Rank-N-Contrast: Learning Continuous Representations for Regression},
    author={Zha, Kaiwen and Cao, Peng and Son, Jeany and Yang, Yuzhe and Katabi, Dina},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023}
}
```
