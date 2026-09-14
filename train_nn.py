

import os
import time
import numpy as np
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from train_common import get_fold_indices, prepare_fold, summarise

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = "./data"
OUTPUT_DIR  = "./models"
RESULTS_DIR = "./results"

HIDDEN_LAYERS  = [256, 128, 64]
DROPOUT_RATE   = 0.3
ACTIVATION     = "relu"

EPOCHS               = 50
BATCH_SIZE           = 1024
LEARNING_RATE        = 0.001
VALIDATION_SPLIT     = 0.10
EARLY_STOP_PATIENCE  = 5

RANDOM_STATE = 42


def build_mlp(input_dim, seed):
    tf.random.set_seed(seed)
    model = keras.Sequential(name="MLP_IDS")
    model.add(layers.Input(shape=(input_dim,)))
    for units in HIDDEN_LAYERS:
        model.add(layers.Dense(units, activation=ACTIVATION))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading preprocessed data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    folds = get_fold_indices(y)
    print(f"\nRunning {len(folds)} repeated stratified folds...\n")

    fold_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "train_time": [], "infer_time": []
    }
    all_cms = []
    last_model = None
    last_history = None
    last_y_test, last_y_pred = None, None

    for i, (train_idx, test_idx) in enumerate(folds, 1):
        X_train, X_test, y_train, y_test = prepare_fold(
            X, y, train_idx, test_idx, apply_smote=True,
            smote_random_state=RANDOM_STATE + i
        )

        model = build_mlp(X_train.shape[1], seed=RANDOM_STATE + i)

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=0
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        )

        start = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        train_time = time.time() - start

        start = time.time()
        y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
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

        n_epochs_ran = len(history.history["loss"])
        print(f"  Fold {i:2d}/{len(folds)} | Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Epochs: {n_epochs_ran} | Train: {train_time:.2f}s")

        last_model = model
        last_history = history.history
        last_y_test, last_y_pred = y_test, y_pred

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MLP — REPEATED K-FOLD RESULTS (mean ± std)")
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

    last_model.save(os.path.join(OUTPUT_DIR, "neural_network.keras"))
    print(f"\nRepresentative model (final fold) saved to '{OUTPUT_DIR}/neural_network.keras'")

    results = {
        "model": "Neural Network (MLP)",
        "summary": summary,
        "cm_summed": cm_sum,
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
        "history": last_history,   # representative training history (final fold)
    }
    joblib.dump(results, os.path.join(RESULTS_DIR, "nn_results.pkl"))
    print(f"Full results saved to '{RESULTS_DIR}/nn_results.pkl'")


if __name__ == "__main__":
    main()
