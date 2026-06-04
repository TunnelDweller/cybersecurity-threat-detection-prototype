"""
preprocess.py
-------------
Loads and preprocesses the CIC-IDS2017 dataset for use in ML and DL models.
Includes SMOTE oversampling to address class imbalance.

Usage:
    python preprocess.py


"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib

#  Configuration 

# Folder containing CIC-IDS2017 CSV files
DATA_DIR = "./cicids2017"

BINARY_CLASSIFICATION = True

SAMPLE_SIZE = 50000

TEST_SIZE = 0.30
RANDOM_STATE = 42

OUTPUT_DIR = "./data"

#  Column name cleaning 

def clean_column_names(df):
    df.columns = df.columns.str.strip()
    return df

#  Load all CSVs

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

#  Preprocessing 

def preprocess(df, binary=True):
    # 1. Identify label column
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

    # 2. Separate features and labels
    X = df.drop(columns=[label_col])
    y = df[label_col].str.strip()

    # 3. Drop non-numeric columns
    cols_to_drop = [c for c in X.columns if X[c].dtype == object]
    if cols_to_drop:
        print(f"\nDropping non-numeric columns: {cols_to_drop}")
        X = X.drop(columns=cols_to_drop)

    # 4. Replace inf values and drop NaN rows
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    before = len(X)
    X.dropna(inplace=True)
    y = y[X.index]
    print(f"\nRemoved {before - len(X):,} rows with NaN/Inf values.")

    # 5. Binary vs multi-class labelling
    if binary:
        y = y.apply(lambda v: 0 if v.upper() == "BENIGN" else 1)
        print("\nBinary labels — 0 = BENIGN, 1 = ATTACK")
    else:
        le = LabelEncoder()
        y = le.fit_transform(y)
        joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))
        print(f"\nMulti-class labels: {list(le.classes_)}")

    # 6. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    feature_names = np.array(X.columns.tolist())

    return X_scaled, np.array(y), feature_names

#  Main 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_dataset(DATA_DIR)

    # Reduced subset — stratified so class proportions are preserved
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    df = df.reset_index(drop=True)
    print(f"\nUsing stratified subset of {len(df):,} records.")

    X, y, feature_names = preprocess(df, binary=BINARY_CLASSIFICATION)

    # Train/test split BEFORE SMOTE — test set must remain unbalanced (real-world)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nClass distribution before SMOTE (training set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = "BENIGN" if u == 0 else "ATTACK"
        print(f"  {label}: {c:,}")

    # SMOTE — applied to training set only
    print("\nApplying SMOTE to balance training set...")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print(f"\nClass distribution after SMOTE (training set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        label = "BENIGN" if u == 0 else "ATTACK"
        print(f"  {label}: {c:,}")

    # Save arrays
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    np.save(os.path.join(OUTPUT_DIR, "feature_names.npy"), feature_names)

    print(f"\nPreprocessing complete.")
    print(f"  Training samples (after SMOTE) : {len(X_train):,}")
    print(f"  Testing samples  (unmodified)  : {len(X_test):,}")
    print(f"  Features                       : {X_train.shape[1]}")
    print(f"\nSaved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
