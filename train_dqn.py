"""
train_dqn.py
------------
Trains a Deep Q-Network (DQN) agent for binary threat classification
on the preprocessed CIC-IDS2017 data.

Optimised for speed — trains in mini-batches rather than sample-by-sample.

Usage:
    python train_dqn.py

Prerequisites:
    Run preprocess.py first to generate the data/ folder.

"""

import os
import time
import random
import numpy as np
import joblib
from collections import deque

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# ── Configuration

DATA_DIR    = "./data"
OUTPUT_DIR  = "./models"
RESULTS_DIR = "./results"

# DQN Hyperparameters
EPISODES        = 10       
BATCH_SIZE      = 512      
GAMMA           = 0.95     
EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPSILON_DECAY   = 0.90     
LEARNING_RATE   = 0.001
MEMORY_SIZE     = 10000   
TARGET_UPDATE   = 2        
STEPS_PER_EP    = 500      

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ── Build Q-Network 

def build_q_network(input_dim, n_actions=2):
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

# ── Replay Buffer 

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

# ── Training 

def train_dqn(X_train, y_train, input_dim):
    q_network      = build_q_network(input_dim)
    target_network = build_q_network(input_dim)
    target_network.set_weights(q_network.get_weights())

    replay_buffer = ReplayBuffer(MEMORY_SIZE)
    epsilon = EPSILON_START
    n_samples = len(X_train)

    print(f"\nTraining DQN for {EPISODES} episodes, {STEPS_PER_EP} steps each...")
    print(f"  Batch size: {BATCH_SIZE}, Memory: {MEMORY_SIZE}, Gamma: {GAMMA}\n")

    start = time.time()

    for episode in range(1, EPISODES + 1):
        indices = np.random.choice(n_samples, STEPS_PER_EP, replace=False)
        correct = 0
        losses  = []

        for idx in indices:
            state      = X_train[idx].astype(np.float32)
            true_label = int(y_train[idx])

            if random.random() < epsilon:
                action = random.randint(0, 1)
            else:
                q_vals = q_network(state[np.newaxis], training=False).numpy()[0]
                action = int(np.argmax(q_vals))

            reward     = 1.0 if action == true_label else -1.0
            correct   += int(action == true_label)

            next_idx   = np.random.randint(0, n_samples)
            next_state = X_train[next_idx].astype(np.float32)

            replay_buffer.push(state, action, reward, next_state)

            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states = replay_buffer.sample(BATCH_SIZE)

                next_q    = target_network(next_states, training=False).numpy()
                target_q  = rewards + GAMMA * np.max(next_q, axis=1)

                with tf.GradientTape() as tape:
                    q_vals_all   = q_network(states, training=True)
                    action_mask  = tf.one_hot(actions, depth=2)
                    q_vals_taken = tf.reduce_sum(q_vals_all * action_mask, axis=1)
                    loss         = tf.reduce_mean(tf.square(target_q - q_vals_taken))

                grads = tape.gradient(loss, q_network.trainable_variables)
                q_network.optimizer.apply_gradients(
                    zip(grads, q_network.trainable_variables)
                )
                losses.append(float(loss))

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_network.set_weights(q_network.get_weights())

        acc      = correct / STEPS_PER_EP
        avg_loss = np.mean(losses) if losses else 0.0
        elapsed  = time.time() - start
        print(f"  Episode {episode:2d}/{EPISODES} | "
              f"Acc: {acc:.4f} | "
              f"Loss: {avg_loss:.4f} | "
              f"Epsilon: {epsilon:.4f} | "
              f"Time: {elapsed:.1f}s")

    train_time = time.time() - start
    print(f"\nTotal training time: {train_time:.2f}s")
    return q_network, train_time

# ── Evaluation

def evaluate_dqn(q_network, X_test, y_test):
    print("\nEvaluating on test set...")
    start  = time.time()

    q_vals = q_network(X_test.astype(np.float32), training=False).numpy()
    y_pred = np.argmax(q_vals, axis=1)

    infer_time = time.time() - start

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n── DQN Results ────────────────────────────────────────")
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

    return acc, prec, rec, f1, cm, y_pred, infer_time

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    input_dim = X_train.shape[1]
    print(f"  Train: {X_train.shape},  Test: {X_test.shape}")

    global train_time
    q_network, train_time = train_dqn(X_train, y_train, input_dim)
    acc, prec, rec, f1, cm, y_pred, infer_time = evaluate_dqn(
        q_network, X_test, y_test
    )

    q_network.save(os.path.join(OUTPUT_DIR, "dqn_model.keras"))
    print(f"\nModel saved to '{OUTPUT_DIR}/dqn_model.keras'")

    results = {
        "model"      : "DQN (Reinforcement Learning)",
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
    }
    joblib.dump(results, os.path.join(RESULTS_DIR, "dqn_results.pkl"))
    print(f"Results saved to '{RESULTS_DIR}/dqn_results.pkl'")

if __name__ == "__main__":
    main()