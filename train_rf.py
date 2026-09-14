

import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from train_common import get_fold_indices, prepare_fold, summarise

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = "./data"
OUTPUT_DIR  = "./models"
RESULTS_DIR = "./results"

N_ESTIMATORS = 100
MAX_DEPTH    = None
N_JOBS       = -1
RANDOM_STATE = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading preprocessed data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))
    feature_names = np.load(
        os.path.join(DATA_DIR, "feature_names.npy"), allow_pickle=True
    )
    print(f"  Full dataset: {X.shape}")

    folds = get_fold_indices(y)
    print(f"\nRunning {len(folds)} repeated stratified folds...\n")

    fold_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "train_time": [], "infer_time": []
    }
    all_cms = []
    all_importances = []
    last_model = None
    last_y_test, last_y_pred = None, None

    for i, (train_idx, test_idx) in enumerate(folds, 1):
        X_train, X_test, y_train, y_test = prepare_fold(
            X, y, train_idx, test_idx, apply_smote=True,
            smote_random_state=RANDOM_STATE + i  # vary slightly per fold
        )

        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE + i
        )

        start = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start

        start = time.time()
        y_pred = rf.predict(X_test)
        infer_time = time.time() - start

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)
        fold_metrics["train_time"].append(train_time)
        fold_metrics["infer_time"].append(infer_time)
        all_cms.append(cm)
        all_importances.append(rf.feature_importances_)

        print(f"  Fold {i:2d}/{len(folds)} | Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Train: {train_time:.2f}s")

        last_model = rf
        last_y_test, last_y_pred = y_test, y_pred

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RANDOM FOREST — REPEATED K-FOLD RESULTS (mean ± std)")
    print("=" * 60)
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "train_time", "infer_time"]:
        m, s = summarise(fold_metrics[key])
        summary[key] = {"mean": m, "std": s, "all": fold_metrics[key]}
        print(f"  {key:<12}: {m:.4f} ± {s:.4f}")
    print("=" * 60)

    cm_sum = np.sum(all_cms, axis=0)
    print("\nSummed confusion matrix across all folds:")
    print(cm_sum)

    mean_importances = np.mean(all_importances, axis=0)
    top_idx = np.argsort(mean_importances)[::-1][:15]
    print("\nTop 15 Feature Importances (averaged across folds):")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<40s}  {mean_importances[idx]:.4f}")

    joblib.dump(last_model, os.path.join(OUTPUT_DIR, "random_forest.pkl"))
    print(f"\nRepresentative model (final fold) saved to '{OUTPUT_DIR}/random_forest.pkl'")

    results = {
        "model": "Random Forest",
        "summary": summary,
        "cm_summed": cm_sum,
        "feature_importances_mean": mean_importances,
        "feature_names": feature_names,
        "n_folds": len(folds),
        "accuracy": summary["accuracy"]["mean"],
        "precision": summary["precision"]["mean"],
        "recall": summary["recall"]["mean"],
        "f1": summary["f1"]["mean"],
        "train_time": summary["train_time"]["mean"],
        "infer_time": summary["infer_time"]["mean"],
        "cm": cm_sum,
        "y_test": last_y_test,
        "y_pred": last_y_pred,
        "feature_importances": mean_importances,
    }
    joblib.dump(results, os.path.join(RESULTS_DIR, "rf_results.pkl"))
    print(f"Full results saved to '{RESULTS_DIR}/rf_results.pkl'")


if __name__ == "__main__":
    main()
