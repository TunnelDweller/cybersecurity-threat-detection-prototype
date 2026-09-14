

import os
import time
import random
import numpy as np
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from train_common import get_fold_indices, prepare_fold, summarise

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR    = "./data"
OUTPUT_DIR  = "./models"
RESULTS_DIR = "./results"

# DQN Hyperparameters
EPISODES        = 20         # moderated from 40 -- see module docstring
STEPS_PER_EP     = 1500       # moderated from 2000 -- see module docstring
# 20 x 1500 = 30,000 step-visits per fold. With ~50,600 SMOTE-balanced
# training rows per fold, this covers a large majority of the fold's rows
# at least once, some more than once -- roughly 6x the original's ~9%
# coverage, while keeping total runtime practical.

FOLD_SUBSET      = 6          # use only the first 6 of the 10 shared folds

BATCH_SIZE      = 512
GAMMA           = 0.95
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.93
LEARNING_RATE   = 0.001
MEMORY_SIZE     = 20000
TARGET_UPDATE   = 2

RANDOM_STATE = 42


def build_q_network(input_dim, seed, n_actions=2):
    tf.random.set_seed(seed)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_actions, activation="linear")
    ], name="DQN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


def make_episode_walk(n_samples, steps_per_ep, rng):
    """
    Builds ONE episode's sequence of row indices as a genuine sequential
    walk: a contiguous block of shuffled indices, taken in order. This
    replaces the original design where `next_state` was an unrelated
    random row -- here, the "next state" the agent sees really is the
    next step of this episode's walk.
    """
    walk = []
    remaining = steps_per_ep
    while remaining > 0:
        order = rng.permutation(n_samples)
        take = min(remaining, n_samples)
        walk.extend(order[:take].tolist())
        remaining -= take
    return walk


def train_dqn_one_fold(X_train, y_train, input_dim, seed):
    q_network      = build_q_network(input_dim, seed)
    target_network = build_q_network(input_dim, seed + 1000)
    target_network.set_weights(q_network.get_weights())

    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPSILON_START
    n_samples = len(X_train)
    rng = np.random.default_rng(seed)

    seen_indices = set()
    start = time.time()

    for episode in range(1, EPISODES + 1):
        walk = make_episode_walk(n_samples, STEPS_PER_EP, rng)
        correct = 0
        losses = []

        for step in range(len(walk) - 1):
            idx      = walk[step]
            next_idx = walk[step + 1]     # REAL next state in this episode's walk

            seen_indices.add(idx)

            state      = X_train[idx].astype(np.float32)
            true_label = int(y_train[idx])
            next_state = X_train[next_idx].astype(np.float32)

            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                q_vals = q_network(state[np.newaxis], training=False).numpy()[0]
                action = int(np.argmax(q_vals))

            reward   = 1.0 if action == true_label else -1.0
            correct += int(action == true_label)

            replay_buffer.push(state, action, reward, next_state)

            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states = replay_buffer.sample(BATCH_SIZE)
                next_q   = target_network(next_states, training=False).numpy()
                target_q = rewards + GAMMA * np.max(next_q, axis=1)

                with tf.GradientTape() as tape:
                    q_vals_all   = q_network(states, training=True)
                    action_mask  = tf.one_hot(actions, depth=2)
                    q_vals_taken = tf.reduce_sum(q_vals_all * action_mask, axis=1)
                    loss = tf.reduce_mean(tf.square(target_q - q_vals_taken))

                grads = tape.gradient(loss, q_network.trainable_variables)
                q_network.optimizer.apply_gradients(
                    zip(grads, q_network.trainable_variables)
                )
                losses.append(float(loss))

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_network.set_weights(q_network.get_weights())

        if episode % 5 == 0 or episode == EPISODES:
            acc = correct / max(1, len(walk) - 1)
            elapsed = time.time() - start
            coverage = len(seen_indices) / n_samples * 100
            print(f"    Episode {episode:3d}/{EPISODES} | Acc: {acc:.4f} | "
                  f"Epsilon: {epsilon:.3f} | Coverage: {coverage:.1f}% | "
                  f"Time: {elapsed:.1f}s")

    train_time = time.time() - start
    final_coverage = len(seen_indices) / n_samples * 100
    print(f"    Final training-row coverage this fold: {final_coverage:.1f}%")
    return q_network, train_time


def evaluate_dqn(q_network, X_test, y_test):
    start = time.time()
    q_vals = q_network(X_test.astype(np.float32), training=False).numpy()
    y_pred = np.argmax(q_vals, axis=1)
    infer_time = time.time() - start

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    return acc, prec, rec, f1, cm, y_pred, infer_time


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading preprocessed data...")
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    all_folds = get_fold_indices(y)         # same 10 folds as RF/MLP
    folds = all_folds[:FOLD_SUBSET]          # DQN uses the first 6 only
    print(f"\nUsing the first {len(folds)} of the {len(all_folds)} shared "
          f"folds (same splits as RF/MLP, for direct comparability).")
    print(f"Each fold: {EPISODES} episodes x {STEPS_PER_EP} steps.")
    print("This is the slowest part of the pipeline -- please be patient.\n")

    fold_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "train_time": [], "infer_time": []
    }
    all_cms = []
    last_model = None
    last_y_test, last_y_pred = None, None

    for i, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n── Fold {i}/{len(folds)} ──────────────────────────")
        X_train, X_test, y_train, y_test = prepare_fold(
            X, y, train_idx, test_idx, apply_smote=True,
            smote_random_state=RANDOM_STATE + i
        )

        q_network, train_time = train_dqn_one_fold(
            X_train, y_train, X_train.shape[1], seed=RANDOM_STATE + i
        )
        acc, prec, rec, f1, cm, y_pred, infer_time = evaluate_dqn(
            q_network, X_test, y_test
        )

        fold_metrics["accuracy"].append(acc)
        fold_metrics["precision"].append(prec)
        fold_metrics["recall"].append(rec)
        fold_metrics["f1"].append(f1)
        fold_metrics["train_time"].append(train_time)
        fold_metrics["infer_time"].append(infer_time)
        all_cms.append(cm)

        print(f"  Fold {i} result -> Acc: {acc:.4f} | F1: {f1:.4f} | "
              f"Train time: {train_time:.2f}s")

        last_model = q_network
        last_y_test, last_y_pred = y_test, y_pred

    # ── Aggregate ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DQN — REPEATED K-FOLD RESULTS (mean ± std)")
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

    last_model.save(os.path.join(OUTPUT_DIR, "dqn_model.keras"))
    print(f"\nRepresentative model (final fold) saved to '{OUTPUT_DIR}/dqn_model.keras'")

    results = {
        "model": "DQN (Reinforcement Learning)",
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
    }
    joblib.dump(results, os.path.join(RESULTS_DIR, "dqn_results.pkl"))
    print(f"Full results saved to '{RESULTS_DIR}/dqn_results.pkl'")


if __name__ == "__main__":
    main()
