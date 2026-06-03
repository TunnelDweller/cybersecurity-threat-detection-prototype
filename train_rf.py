"""
train_rf.py
-----------
Trains a Random Forest classifier on the preprocessed CIC-IDS2017 data.

Usage:
    python train_rf.py

Prerequisites:
    Run preprocess.py first to generate the data/ folder.

"""

import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Configuration 

DATA_DIR   = "./data"
OUTPUT_DIR = "./models"
RESULTS_DIR = "./results"

# Random Forest hyperparameters
N_ESTIMATORS = 100
MAX_DEPTH    = None      # None = grow full trees
N_JOBS       = -1        # Use all CPU cores
RANDOM_STATE = 42

# ── Main 

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    feature_names = np.load(
        os.path.join(DATA_DIR, "feature_names.npy"), allow_pickle=True
    )

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── Train
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE
    )

    start = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.2f}s")

    # ── Evaluate 
    print("\nEvaluating on test set...")
    start = time.time()
    y_pred = rf.predict(X_test)
    infer_time = time.time() - start

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n── Random Forest Results ──────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Train time: {train_time:.2f}s")
    print(f"  Infer time: {infer_time:.4f}s ({infer_time/len(X_test)*1000:.4f} ms/sample)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(cm)

    # ── Feature Importance (top 15) 
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 Feature Importances:")
    for rank, i in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[i]:<40s}  {importances[i]:.4f}")

    # ── Save model and results 
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "random_forest.pkl"))
    print(f"\nModel saved to '{OUTPUT_DIR}/random_forest.pkl'")

    results = {
        "model"      : "Random Forest",
        "accuracy"   : acc,
        "precision"  : prec,
        "recall"     : rec,
        "f1"         : f1,
        "train_time" : train_time,
        "infer_time" : infer_time,
        "n_test"     : len(X_test),
        "cm"         : cm,
        "y_test"     : y_test,
        "y_pred"     : y_pred,
        "feature_importances": importances,
        "feature_names"      : feature_names,
    }
    joblib.dump(results, os.path.join(RESULTS_DIR, "rf_results.pkl"))
    print(f"Results saved to '{RESULTS_DIR}/rf_results.pkl'")

if __name__ == "__main__":
    main()
