

import os
import glob
import numpy as np
import pandas as pd



DATA_DIR     = "./cicids2017"     
SAMPLE_SIZE  = 50000
RANDOM_STATE = 42
OUTPUT_DIR   = "./data"


def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df


def load_dataset(data_dir):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Please update DATA_DIR to point to your CIC-IDS2017 folder."
        )

    print(f"Found {len(csv_files)} CSV file(s). Loading...")
    frames = []
    for f in csv_files:
        print(f"  Loading: {os.path.basename(f)}")
        df = pd.read_csv(f, low_memory=False)
        df = clean_column_names(df)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal records loaded: {len(combined):,}")
    return combined


def clean_and_encode(df):
    """
    Cleans the raw dataframe and returns numeric features (unscaled) plus
    binary labels. Scaling is deliberately NOT done here -- it happens
    per-fold in train_common.py to avoid any leakage.
    """
    label_col = None
    for candidate in ["Label", " Label", "label"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        raise ValueError("Could not find a 'Label' column in the dataset.")

    print(f"\nUsing label column: '{label_col}'")
    print("Class distribution (raw):")
    print(df[label_col].value_counts())

    X = df.drop(columns=[label_col])
    y = df[label_col].str.strip()

    cols_to_drop = [c for c in X.columns if X[c].dtype == object]
    if cols_to_drop:
        print(f"\nDropping non-numeric columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    X = X.replace([np.inf, -np.inf], np.nan)
    before = len(X)
    X = X.dropna()
    y = y[X.index]
    print(f"\nRemoved {before - len(X):,} rows with NaN/Inf values.")

    # Binary labels: 0 = BENIGN, 1 = ATTACK
    y_bin = y.apply(lambda v: 0 if v.upper() == "BENIGN" else 1)
    print("\nBinary labels — 0 = BENIGN, 1 = ATTACK")

    feature_names = np.array(X.columns.tolist())
    return X.reset_index(drop=True).values.astype(np.float64), \
           np.array(y_bin.values), \
           feature_names


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset(DATA_DIR)

    # Stratified subset -- preserves class proportions of the full dataset
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    print(f"\nUsing stratified subset of {len(df):,} records.")

    X, y, feature_names = clean_and_encode(df)

    print(f"\nClass distribution (full 50,000-record subset):")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        label = "BENIGN" if u == 0 else "ATTACK"
        print(f"  {label}: {c:,} ({c/len(y)*100:.2f}%)")

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, "feature_names.npy"), feature_names)

    print(f"\nPreprocessing complete.")
    print(f"  Total samples : {len(X):,}")
    print(f"  Features      : {X.shape[1]}")
    print(f"\nSaved to '{OUTPUT_DIR}/' as X.npy, y.npy, feature_names.npy")
    print("\nNOTE: scaling and SMOTE are now applied per-fold at training")
    print("time (see train_common.py) rather than once here, so that every")
    print("model's cross-validation is leakage-free.")


if __name__ == "__main__":
    main()
