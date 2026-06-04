"""
generate_additional_plots.py
----------------------------
Generates additional evaluation plots for the thesis:
1. ROC Curve (all three models)
2. Precision-Recall Curve (all three models)
3. False Positive vs False Negative comparison
4. Learning Curve (Random Forest)

Usage:
    python generate_additional_plots.py

Prerequisites:
    Run train_rf.py, train_nn.py and train_dqn.py first.


"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve

RESULTS_DIR = "./results"
PLOTS_DIR   = "./plots"
DATA_DIR    = "./data"
MODELS_DIR  = "./models"

os.makedirs(PLOTS_DIR, exist_ok=True)

#  Load results 

print("Loading results...")
rf_res  = joblib.load(os.path.join(RESULTS_DIR, "rf_results.pkl"))
nn_res  = joblib.load(os.path.join(RESULTS_DIR, "nn_results.pkl"))
dqn_res = joblib.load(os.path.join(RESULTS_DIR, "dqn_results.pkl"))

X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))

#  Load models for probability scores

print("Loading models...")
rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
nn_model  = tf.keras.models.load_model(os.path.join(MODELS_DIR, "neural_network.keras"))
dqn_model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "dqn_model.keras"))

#  Get probability scores

print("Getting probability scores...")

# RF - probability of attack class
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# MLP - sigmoid output
nn_probs = nn_model.predict(X_test, batch_size=1024, verbose=0).flatten()

# DQN - softmax Q-values, use attack class probability
dqn_qvals = dqn_model.predict(X_test.astype(np.float32), batch_size=1024, verbose=0)
dqn_probs = tf.nn.softmax(dqn_qvals).numpy()[:, 1]

#  Plot 1: ROC Curve

print("Generating ROC curve...")
fig, ax = plt.subplots(figsize=(8, 6))

colors = ["#2196F3", "#FF5722", "#4CAF50"]
models = [
    ("Random Forest",          rf_probs),
    ("Neural Network (MLP)",   nn_probs),
    ("DQN (Reinforcement Learning)", dqn_probs),
]

for (name, probs), color in zip(models, colors):
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {roc_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
plt.close()
print("  Saved: plots/roc_curve.png")

#  Plot 2: Precision-Recall Curve

print("Generating Precision-Recall curve...")
fig, ax = plt.subplots(figsize=(8, 6))

for (name, probs), color in zip(models, colors):
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f"{name} (AUC = {pr_auc:.4f})")

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curve Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curve.png"), dpi=150)
plt.close()
print("  Saved: plots/precision_recall_curve.png")

#  Plot 3: False Positive vs False Negative

print("Generating FP vs FN comparison...")
model_names = ["Random Forest", "Neural Network\n(MLP)", "DQN (RL)"]

def get_fp_fn(cm):
    fp = cm[0][1]  # benign predicted as attack
    fn = cm[1][0]  # attack predicted as benign
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
ax.set_ylabel("Count", fontsize=12)
ax.set_title("False Positives and False Negatives by Model", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fp_fn_comparison.png"), dpi=150)
plt.close()
print("  Saved: plots/fp_fn_comparison.png")

#  Plot 4: Learning Curve (RF)

print("Generating RF learning curve (this may take a few minutes)...")
rf_lc = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    rf_lc, X_train, y_train,
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
ax.set_title("Random Forest Learning Curve", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "learning_curve_rf.png"), dpi=150)
plt.close()
print("  Saved: plots/learning_curve_rf.png")

print("\nAll additional plots saved to 'plots/'")
print("Done.")
