"""
train_nn.py
-----------
Trains a Multi-Layer Perceptron (MLP / feedforward neural network) on the
preprocessed CIC-IDS2017 data using TensorFlow/Keras.

Usage:
    python train_nn.py

Prerequisites:
    pip install tensorflow
    Run preprocess.py first to generate the data/ folder.

"""

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
    f1_score, confusion_matrix, classification_report
)

#  Configuration 

DATA_DIR    = "./data"
OUTPUT_DIR  = "./models"
RESULTS_DIR = "./results"

# MLP Architecture
HIDDEN_LAYERS  = [256, 128, 64]   
DROPOUT_RATE   = 0.3
ACTIVATION     = "relu"

# Training
EPOCHS         = 50
BATCH_SIZE     = 1024
LEARNING_RATE  = 0.001
VALIDATION_SPLIT = 0.10            
EARLY_STOP_PATIENCE = 5            

RANDOM_STATE = 42
tf.random.set_seed(RANDOM_STATE)

#  Build model 

def build_mlp(input_dim, num_classes):
    model = keras.Sequential(name="MLP_IDS")

    # Input layer
    model.add(layers.Input(shape=(input_dim,)))

    # Hidden layers
    for units in HIDDEN_LAYERS:
        model.add(layers.Dense(units, activation=ACTIVATION))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(DROPOUT_RATE))

    # Output layer
    if num_classes == 2:
        model.add(layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    else:
        model.add(layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"
        metrics = ["accuracy"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=metrics
    )
    return model, loss

#  Main 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    num_classes = len(np.unique(y_train))
    input_dim   = X_train.shape[1]

    print(f"  Train: {X_train.shape},  Test: {X_test.shape}")
    print(f"  Classes: {num_classes},  Features: {input_dim}")

    #  Build 
    model, loss_fn = build_mlp(input_dim, num_classes)
    model.summary()

    #  Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )

    #  Train 
    print("\nTraining Neural Network...")
    start = time.time()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    train_time = time.time() - start
    print(f"\nTraining time: {train_time:.2f}s")

    #  Evaluate 
    print("\nEvaluating on test set...")
    start = time.time()

    if num_classes == 2:
        y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)

    infer_time = time.time() - start

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

    print("\n Neural Network (MLP) Results ─")
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

    #  Save model and results 
    model.save(os.path.join(OUTPUT_DIR, "neural_network.keras"))
    print(f"\nModel saved to '{OUTPUT_DIR}/neural_network.keras'")

    results = {
        "model"      : "Neural Network (MLP)",
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
        "history"    : history.history,
    }
    joblib.dump(results, os.path.join(RESULTS_DIR, "nn_results.pkl"))
    print(f"Results saved to '{RESULTS_DIR}/nn_results.pkl'")

if __name__ == "__main__":
    main()
