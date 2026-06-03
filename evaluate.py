"""
evaluate.py
-----------
Loads saved results from all three models, prints a side-by-side comparison,
and generates plots.

Usage:
    python evaluate.py

Prerequisites:
    Run train_rf.py, train_nn.py, and train_dqn.py first.

"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

RESULTS_DIR = "./results"
PLOTS_DIR   = "./plots"

# ── Helpers 

def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: '{path}'. "
            "Make sure you have run the corresponding training script."
        )
    return joblib.load(path)


def plot_confusion_matrix(cm, model_name, save_path, class_names=None):
    if class_names is None:
        class_names = ["BENIGN", "ATTACK"] if cm.shape[0] == 2 else [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    acc_key     = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    axes[1].plot(history[acc_key], label="Train Acc", linewidth=2)
    axes[1].plot(history[val_acc_key], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_title("Training & Validation Accuracy", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(importances, feature_names, save_path, top_n=15):
    top_idx  = np.argsort(importances)[::-1][:top_n]
    top_imp  = importances[top_idx]
    top_names = feature_names[top_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))[::-1]
    ax.barh(range(top_n), top_imp[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results_list, save_path):
    """Bar chart comparing all three models across four metrics."""
    metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    keys      = ["accuracy", "precision", "recall", "f1"]
    colors    = ["#2196F3", "#FF5722", "#4CAF50"]
    x         = np.arange(len(metrics))
    n_models  = len(results_list)
    width     = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, res in enumerate(results_list):
        vals = [res[k] for k in keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=res["model"], color=colors[i], alpha=0.87)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: RF vs MLP vs DQN",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_train_time_comparison(results_list, save_path):
    """Horizontal bar chart comparing training times."""
    names  = [r["model"] for r in results_list]
    times  = [r["train_time"] for r in results_list]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, times, color=colors, alpha=0.87)
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{t:.2f}s", va="center", fontsize=10)
    ax.set_xlabel("Training Time (seconds)", fontsize=11)
    ax.set_title("Training Time Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ── Main 

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading results...")
    rf_res  = load_results("rf_results.pkl")
    nn_res  = load_results("nn_results.pkl")
    dqn_res = load_results("dqn_results.pkl")

    all_results = [rf_res, nn_res, dqn_res]

    # ── Summary table 
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 72)
    print(f"{'Metric':<20} {'Random Forest':>16} {'Neural Network':>16} {'DQN (RL)':>12}")
    print("-" * 72)

    rows = [
        ("Accuracy",       "accuracy"),
        ("Precision",      "precision"),
        ("Recall",         "recall"),
        ("F1-Score",       "f1"),
        ("Train Time (s)", "train_time"),
        ("Infer Time (s)", "infer_time"),
    ]
    for name, key in rows:
        rv  = rf_res[key]
        nv  = nn_res[key]
        dv  = dqn_res[key]
        print(f"  {name:<18} {rv:>16.4f} {nv:>16.4f} {dv:>12.4f}")
    print("=" * 72)

    # ── Plots 
    print("\nGenerating plots...")

    plot_confusion_matrix(
        rf_res["cm"], "Random Forest",
        os.path.join(PLOTS_DIR, "confusion_matrix_rf.png")
    )
    plot_confusion_matrix(
        nn_res["cm"], "Neural Network (MLP)",
        os.path.join(PLOTS_DIR, "confusion_matrix_nn.png")
    )
    plot_confusion_matrix(
        dqn_res["cm"], "DQN (Reinforcement Learning)",
        os.path.join(PLOTS_DIR, "confusion_matrix_dqn.png")
    )

    if "history" in nn_res:
        plot_training_history(
            nn_res["history"],
            os.path.join(PLOTS_DIR, "training_history_nn.png")
        )

    if "feature_importances" in rf_res:
        plot_feature_importance(
            rf_res["feature_importances"],
            rf_res["feature_names"],
            os.path.join(PLOTS_DIR, "feature_importance_rf.png")
        )

    plot_model_comparison(
        all_results,
        os.path.join(PLOTS_DIR, "model_comparison.png")
    )

    plot_train_time_comparison(
        all_results,
        os.path.join(PLOTS_DIR, "train_time_comparison.png")
    )

    print(f"\nAll plots saved to '{PLOTS_DIR}/'")
    print("Done.")

if __name__ == "__main__":
    main()