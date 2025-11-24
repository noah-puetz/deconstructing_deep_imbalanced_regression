import pandas as pd
import numpy as np
import os
import collections
import random

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

def load_age_data(file_path):
    df = pd.read_csv(file_path)
    df['split'] = df['split'].astype(str)
    return df

def resample_age_data(weights, df, max_test_samples=50, test_cutoff=None, align_train_and_test=False):
    sampled_dfs = []
    od = collections.OrderedDict(sorted(weights.items()))
    total_weight = sum(od.values())
    for age, weight in od.items():
        group = df[df['age'] == age]
        n_samples = max(0, int((len(df)*4) * weight / total_weight))
        if len(group) > 0:
            sampled = group.sample(min(len(group), n_samples), replace=True)
            sampled_dfs.append(sampled)

    df_train = pd.concat(sampled_dfs)
    df_train['split'] = 'train'
    df_test= df[~df.index.isin(df_train.index)].copy()
    df_test['split'] = 'test'
    
    sampled_dfs = []
    for age, group_df in df_test.groupby('age'):
        df_group = group_df.sample(min(len(group_df), max_test_samples), replace=True)
        if test_cutoff == "uniform":
            if len(df_group) < max_test_samples:
                continue 
        sampled_dfs.append(df_group)
    df_test = pd.concat(sampled_dfs)
    
    if isinstance(test_cutoff, tuple) or isinstance(test_cutoff, list):
        assert len(test_cutoff) == 2, "test_cutoff should be a tuple or list of length 2"
        df_test = df_test[df_test['age'] >= test_cutoff[0]]
        df_test = df_test[df_test['age'] <= test_cutoff[1]]
        df_test['split'] = 'test'
        
    if align_train_and_test:    
        df_train = df_train[df_train['age'] >= df_test['age'].min()]
        df_train = df_train[df_train['age'] <= df_test['age'].max()]  

    df_val = df_train.sample(frac=0.1, random_state=45)
    df_val['split'] = 'val'
    df_train = df_train[~df_train.index.isin(df_val.index)]
    return df_train, df_val, df_test

def main():
    seed_everything(42)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    agedb_path = os.path.join(base_dir, "agedb.csv")
    
    if not os.path.exists(agedb_path):
        print(f"Error: {agedb_path} not found.")
        return

    df = load_age_data(agedb_path)
    
    print("Generating Extrapolation Protocol...")
    ages = df['age'].unique()
    age_weights = {}
    for age in ages:
        peak1 = np.exp(-0.01 * (age - 50)**2)
        age_weights[age] = peak1
        
    df_train_small, df_val_small, df_test_small = resample_age_data(age_weights, df, test_cutoff="uniform")
    pd.concat([df_train_small, df_val_small, df_test_small]).to_csv(os.path.join(base_dir, "agedb_extrapolation.csv"), index=False)
    print(f"Saved to {os.path.join(base_dir, 'agedb_extrapolation.csv')}")

    print("Generating Interpolation Protocol...")
    ages = df['age'].unique()
    age_weights = {}
    for age in ages:
        peak1 = np.exp(-0.01 * (age - 25)**2)
        peak2 = np.exp(-0.01 * (age - 80)**2)
        age_weights[age] = peak1 + peak2

    df_train_tp, df_val_tp, df_test_tp = resample_age_data(age_weights, df, test_cutoff="uniform", align_train_and_test=True)
    pd.concat([df_train_tp, df_val_tp, df_test_tp]).to_csv(os.path.join(base_dir, "agedb_interpolation.csv"), index=False)
    print(f"Saved to {os.path.join(base_dir, 'agedb_interpolation.csv')}")

    print("Generating Blindspot Protocol...")
    df_filtered = df[(df['age'] >= 20) & (df['age'] <= 73)]
    ages = df_filtered['age'].unique()
    age_weights = {age: 1 for age in ages}

    df_train_hard, df_val_hard, df_test_hard = resample_age_data(age_weights, df_filtered, max_test_samples=50, align_train_and_test=True)

    df_train_hard = df_train_hard[(df_train_hard['age'] < 47) | (df_train_hard['age'] > 53)]
    df_val_hard = df_val_hard[(df_val_hard['age'] < 47) | (df_val_hard['age'] > 53)]

    pd.concat([df_train_hard, df_val_hard, df_test_hard]).to_csv(os.path.join(base_dir, "agedb_blindspot.csv"), index=False)
    print(f"Saved to {os.path.join(base_dir, 'agedb_blindspot.csv')}")

if __name__ == "__main__":
    main()
