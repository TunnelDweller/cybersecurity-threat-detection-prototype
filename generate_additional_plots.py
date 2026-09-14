

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

from train_common import get_fold_indices, prepare_fold

RESULTS_DIR = "./results"
PLOTS_DIR   = "./plots"
DATA_DIR    = "./data"
MODELS_DIR  = "./models"

RANDOM_STATE = 42

os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load results (for summed confusion matrices) ────────────────────────────

print("Loading results...")
rf_res  = joblib.load(os.path.join(RESULTS_DIR, "rf_results.pkl"))
nn_res  = joblib.load(os.path.join(RESULTS_DIR, "nn_results.pkl"))
dqn_res = joblib.load(os.path.join(RESULTS_DIR, "dqn_results.pkl"))

print("Loading full dataset and rebuilding fold splits...")
X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
folds = get_fold_indices(y)

# NOTE: the saved models (random_forest.pkl, neural_network.keras,
# dqn_model.keras) are each from their LAST trained fold (fold 10 for
# RF/MLP, fold 6 for DQN) -- not fold 0. We reconstruct that same last
# fold here so the probabilities are generated on the correct
# corresponding held-out test set for each saved model.
rf_last_fold_idx  = rf_res["n_folds"] - 1
nn_last_fold_idx  = nn_res["n_folds"] - 1
dqn_last_fold_idx = dqn_res["n_folds"] - 1


def get_test_split(fold_idx):
    train_idx, test_idx = folds[fold_idx]
    X_train, X_test, y_train, y_test = prepare_fold(
        X, y, train_idx, test_idx, apply_smote=True,
        smote_random_state=RANDOM_STATE + fold_idx + 1
    )
    return X_test, y_test


X_test_rf,  y_test_rf  = get_test_split(rf_last_fold_idx)
X_test_nn,  y_test_nn  = get_test_split(nn_last_fold_idx)
X_test_dqn, y_test_dqn = get_test_split(dqn_last_fold_idx)

# ── Load models for probability scores ──────────────────────────────────────

print("Loading models...")
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
nn_model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "neural_network.keras"))
dqn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "dqn_model.keras"))

# ── Get probability scores ───────────────────────────────────────────────

print("Getting probability scores...")

rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
nn_probs = nn_model.predict(X_test_nn, batch_size=1024, verbose=0).flatten()
dqn_qvals = dqn_model.predict(X_test_dqn.astype(np.float32), batch_size=1024, verbose=0)
dqn_probs = tf.nn.softmax(dqn_qvals).numpy()[:, 1]

# ── Plot 1: ROC Curve ────────────────────────────────────────────────────────

print("Generating ROC curve...")
fig, ax = plt.subplots(figsize=(8, 6))

colors = ["#2196F3", "#FF5722", "#4CAF50"]
models = [
    ("Random Forest",          rf_probs, y_test_rf),
    ("Neural Network (MLP)",   nn_probs, y_test_nn),
    ("DQN (Reinforcement Learning)", dqn_probs, y_test_dqn),
]

for (name, probs, y_true), color in zip(models, colors):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison\n(single representative fold per model)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("  Saved: plots/roc_curve.png")

# ── Plot 2: Precision-Recall Curve ───────────────────────────────────────────

print("Generating Precision-Recall curve...")
fig, ax = plt.subplots(figsize=(8, 6))

for (name, probs, y_true), color in zip(models, colors):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f"{name} (AUC = {pr_auc:.4f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve Comparison\n(single representative fold per model)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()
print("  Saved: plots/precision_recall_curve.png")

# ── Plot 3: False Positive vs False Negative (from SUMMED confusion matrices) ─

print("Generating FP vs FN comparison (summed across all folds)...")
model_names = [f"Random Forest\n(n={rf_res['n_folds']} folds)",
               f"Neural Network\n(n={nn_res['n_folds']} folds)",
               f"DQN (RL)\n(n={dqn_res['n_folds']} folds)"]


def get_fp_fn(cm):
    fp = cm[0][1]
    fn = cm[1][0]
    return fp, fn


rf_fp,  rf_fn  = get_fp_fn(rf_res["cm"])
nn_fp,  nn_fn  = get_fp_fn(nn_res["cm"])
dqn_fp, dqn_fn = get_fp_fn(dqn_res["cm"])

fp_vals = [rf_fp,  nn_fp,  dqn_fp]
fn_vals = [rf_fn,  nn_fn,  dqn_fn]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - width/2, fp_vals, width, label="False Positives", color="#FF5722", alpha=0.85)
bars2 = ax.bar(x + width/2, fn_vals, width, label="False Negatives", color="#2196F3", alpha=0.85)

for bar in list(bars1) + list(bars2):
    height = bar.get_height()
    ax.annotate(str(int(height)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Count (summed across all folds)", fontsize=12)
ax.set_title("False Positives and False Negatives by Model\n(summed across all evaluated folds)",
             fontsize=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fp_fn_comparison.png"), dpi=150)
plt.close()
print("  Saved: plots/fp_fn_comparison.png")

# ── Plot 4: Learning Curve (RF, on full dataset) ─────────────────────────────

print("Generating RF learning curve (this may take a few minutes)...")
rf_lc = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=RANDOM_STATE)

train_sizes, train_scores, val_scores = learning_curve(
    rf_lc, X, y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1
)

train_mean = np.mean(train_scores, axis=1)
train_std  = np.std(train_scores, axis=1)
val_mean   = np.mean(val_scores, axis=1)
val_std    = np.std(val_scores, axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_mean, "o-", color="#2196F3", linewidth=2, label="Training Score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2196F3")
ax.plot(train_sizes, val_mean, "o-", color="#FF5722", linewidth=2, linestyle="--", label="Validation Score")
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#FF5722")

ax.set_xlabel("Training Set Size", fontsize=12)
ax.set_ylabel("F1-Score", fontsize=12)
ax.set_title("Random Forest Learning Curve\n(3-fold CV on full dataset)",
             fontsize=12, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "learning_curve_rf.png"), dpi=150)
plt.close()
print("  Saved: plots/learning_curve_rf.png")

print("\nAll additional plots saved to 'plots/'")
print("Done.")
