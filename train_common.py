

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

N_SPLITS  = 5
N_REPEATS = 2          # 5 x 2 = 10 total train/test folds
RANDOM_STATE = 42


def get_fold_indices(y, n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE):
    """
    Returns a list of (train_idx, test_idx) tuples -- one per fold.
    This is computed ONCE and reused by every model so that RF, MLP and DQN
    are always evaluated on identical held-out rows within each fold index.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    return list(rskf.split(np.zeros(len(y)), y))


def prepare_fold(X, y, train_idx, test_idx, apply_smote=True, smote_random_state=42):
    """
    Given raw (unscaled) X, y and a train/test index pair for ONE fold:
        1. Splits X, y using the provided indices.
        2. Fits StandardScaler on the TRAINING portion only, applies it to
           both train and test (no leakage -- test statistics never seen).
        3. Optionally applies SMOTE to the training portion only, fit fresh
           for this fold. The test portion is NEVER touched by SMOTE, so it
           always reflects the real-world class distribution.

    Returns: X_train, X_test, y_train, y_test  (all numpy arrays, ready to
    train on directly)
    """
    X_train_raw, X_test_raw = X[train_idx], X[test_idx]
    y_train_raw, y_test_raw = y[train_idx], y[test_idx]

    # Scale -- fit on training fold only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test  = scaler.transform(X_test_raw)

    y_train = y_train_raw.copy()
    y_test  = y_test_raw.copy()

    if apply_smote:
        smote = SMOTE(random_state=smote_random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test


def summarise(values):
    """Returns (mean, std) for a list/array of per-fold metric values."""
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))
