# AgeDB-DIR Data

This folder contains the data and scripts for the AgeDB-DIR dataset.

#### Prerequisites

1. Download AgeDB dataset from [here](https://ibug.doc.ic.ac.uk/resources/agedb/) and extract the zip file (you may need to contact the authors of AgeDB dataset for the zip password) to folder `./data`

2. We use the standard train/val/test split file (`agedb.csv` in folder `./data`) provided by Yang et al.(ICML 2021), which is used to set up balanced val/test set. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_agedb.py
python data/preprocess_agedb.py
```

#### Additional Protocols

We introduce three new experiment protocols to isolate different characteristics of DIR:

1.  **Extrapolation**: Training data is concentrated around a specific age (50), testing generalization to unseen ages.
2.  **Interpolation**: Training data has a bimodal distribution (peaks at 25 and 80), testing interpolation capabilities.
3.  **Blindspot**: Training data has a gap in the middle (ages 47-53 removed), testing performance on missing data regions.

To generate the CSV files for these protocols (`agedb_extrapolation.csv`, `agedb_interpolation.csv`, `agedb_blindspot.csv`), run the Python script:

```bash
python data/create_protocols.py
```

This script will generate the CSV files in the `data/` directory.
