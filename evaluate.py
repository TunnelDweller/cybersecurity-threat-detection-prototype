
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

RESULTS_DIR = "./results"
PLOTS_DIR   = "./plots"

MODEL_COLORS = {"Random Forest": "#2196F3", "Neural Network (MLP)": "#FF5722",
                "DQN (Reinforcement Learning)": "#4CAF50"}


def load_results(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Results file not found: '{path}'. "
            "Make sure you have run the corresponding training script."
        )
    return joblib.load(path)


def plot_confusion_matrix(cm, model_name, n_folds, save_path, class_names=None):
    if class_names is None:
        class_names = ["BENIGN", "ATTACK"] if cm.shape[0] == 2 else [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}\n(summed across {n_folds} folds)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    axes[0].set_title("Training & Validation Loss\n(representative fold)", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    acc_key     = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    axes[1].plot(history[acc_key], label="Train Acc", linewidth=2)
    axes[1].plot(history[val_acc_key], label="Val Acc", linewidth=2, linestyle="--")
    axes[1].set_title("Training & Validation Accuracy\n(representative fold)", fontweight="bold")
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
    ax.set_xlabel("Importance Score (mean across folds)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest\n(averaged across all folds)",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results_list, save_path):
    """Bar chart comparing all three models across four metrics, WITH error
    bars showing standard deviation across folds."""
    metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
    keys      = ["accuracy", "precision", "recall", "f1"]
    colors    = ["#2196F3", "#FF5722", "#4CAF50"]
    x         = np.arange(len(metrics))
    n_models  = len(results_list)
    width     = 0.22

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, res in enumerate(results_list):
        means = [res["summary"][k]["mean"] for k in keys]
        stds  = [res["summary"][k]["std"] for k in keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=4,
                      label=f"{res['model']} (n={res['n_folds']})",
                      color=colors[i], alpha=0.87)
        for bar, m in zip(bars, means):
            ax.annotate(
                f"{m:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=7.5
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score (mean ± std across folds)", fontsize=12)
    ax.set_title("Model Comparison: RF vs MLP vs DQN\n(repeated stratified k-fold)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_train_time_comparison(results_list, save_path):
    """Horizontal bar chart comparing mean training times, with error bars.
    Uses a log scale since DQN's training time is roughly 3 orders of
    magnitude larger than RF's, which would make a linear scale unreadable."""
    names  = [f"{r['model']} (n={r['n_folds']})" for r in results_list]
    means  = [r["summary"]["train_time"]["mean"] for r in results_list]
    stds   = [r["summary"]["train_time"]["std"] for r in results_list]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(names, means, xerr=stds, capsize=4, color=colors, alpha=0.87)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height() / 2,
                f"{m:.2f}s ± {s:.2f}", va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Mean Training Time, log scale (seconds)", fontsize=11)
    ax.set_title("Training Time Comparison (mean ± std across folds)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_fold_variance(results_list, save_path):
    """
    NEW FIGURE: strip/scatter plot showing every individual fold's accuracy
    and F1 for every model, alongside the mean. This is the direct visual
    evidence of repeated-run evaluation -- addresses the "n=1, no rigor"
    feedback by showing the actual spread, not just a mean +/- std number.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes, ["accuracy", "f1"], ["Accuracy", "F1-Score"]):
        for i, res in enumerate(results_list):
            vals = res["summary"][metric]["all"]
            n = len(vals)
            xs = np.full(n, i) + np.random.uniform(-0.08, 0.08, size=n)  
            ax.scatter(xs, vals, color=MODEL_COLORS.get(res["model"], "#888"),
                       alpha=0.75, s=50, zorder=3, edgecolor="white", linewidth=0.5)
            mean_val = res["summary"][metric]["mean"]
            ax.hlines(mean_val, i - 0.2, i + 0.2, color="black", linewidth=2, zorder=4)

        ax.set_xticks(range(len(results_list)))
        ax.set_xticklabels([f"{r['model']}\n(n={r['n_folds']})" for r in results_list], fontsize=9)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"Per-fold {title} — individual folds shown as points,\nmean shown as black line",
                     fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading results...")
    rf_res  = load_results("rf_results.pkl")
    nn_res  = load_results("nn_results.pkl")
    dqn_res = load_results("dqn_results.pkl")

    all_results = [rf_res, nn_res, dqn_res]

    print("\n" + "=" * 90)
    print("  MODEL COMPARISON SUMMARY (mean ± std across folds)")
    print("=" * 90)
    print(f"{'Metric':<18} {'Random Forest (n=10)':<24} {'MLP (n=10)':<24} {'DQN (n=6)':<24}")
    print("-" * 90)

    rows = [
        ("Accuracy",       "accuracy"),
        ("Precision",      "precision"),
        ("Recall",         "recall"),
        ("F1-Score",       "f1"),
        ("Train Time (s)", "train_time"),
        ("Infer Time (s)", "infer_time"),
    ]
    for name, key in rows:
        rv  = rf_res["summary"][key]
        nv  = nn_res["summary"][key]
        dv  = dqn_res["summary"][key]
        print(f"  {name:<16} {rv['mean']:>8.4f} ± {rv['std']:<8.4f}    "
              f"{nv['mean']:>8.4f} ± {nv['std']:<8.4f}    "
              f"{dv['mean']:>8.4f} ± {dv['std']:<8.4f}")
    print("=" * 90)


    print("\nGenerating plots...")

    plot_confusion_matrix(
        rf_res["cm"], "Random Forest", rf_res["n_folds"],
        os.path.join(PLOTS_DIR, "confusion_matrix_rf.png")
    )
    plot_confusion_matrix(
        nn_res["cm"], "Neural Network (MLP)", nn_res["n_folds"],
        os.path.join(PLOTS_DIR, "confusion_matrix_nn.png")
    )
    plot_confusion_matrix(
        dqn_res["cm"], "DQN (Reinforcement Learning)", dqn_res["n_folds"],
        os.path.join(PLOTS_DIR, "confusion_matrix_dqn.png")
    )

    if "history" in nn_res and nn_res["history"] is not None:
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

    plot_fold_variance(
        all_results,
        os.path.join(PLOTS_DIR, "fold_variance.png")
    )

    print(f"\nAll plots saved to '{PLOTS_DIR}/'")
    print("Done.")


if __name__ == "__main__":
    main()
