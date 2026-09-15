"""Train the four TinyEdgeBench methods on the shared split and score them.

Methods
    rule      - hand-written thresholds (training/rules.py, no training)
    logistic  - StandardScaler + LogisticRegression
    tree      - DecisionTreeClassifier, depth limited
    mlp       - StandardScaler + MLPClassifier 8 -> 8 -> 4

All methods see exactly the same features and the same train/val/test rows.

Run:
    python training/train.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.generate import FEATURE_COLUMNS, LABELS  # noqa: E402
from training.rules import predict_rule_batch  # noqa: E402

DATA_PATH = ROOT / "dataset" / "data.csv"
SEED = 42

METHODS = ("rule", "logistic", "tree", "mlp")
TREE_MAX_DEPTH = 5
MLP_HIDDEN_LAYERS = (8, 8)
MLP_MAX_ITER = 500

BYTES_PER_FLOAT = 4


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_data(path=DATA_PATH):
    """Load the dataset and return {split: (X, y)} with y as class ids."""
    data = pd.read_csv(path)
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    splits = {}
    for split in ("train", "val", "test"):
        part = data[data["split"] == split]
        splits[split] = (
            part[FEATURE_COLUMNS].to_numpy(dtype=float),
            part["label"].map(label_to_id).to_numpy(dtype=int),
        )
    return splits


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_logistic():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])


def train_tree():
    return DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=SEED)


def train_mlp():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation="relu",
            max_iter=MLP_MAX_ITER,
            random_state=SEED,
        )),
    ])


def train_models(X_train, y_train):
    """Fit every method that needs fitting. `rule` has nothing to fit."""
    models = {
        "rule": None,
        "logistic": train_logistic(),
        "tree": train_tree(),
        "mlp": train_mlp(),
    }
    for name, model in models.items():
        if model is not None:
            model.fit(X_train, y_train)
    return models


def predict_ids(name, model, X):
    """Return predicted class ids for any of the four methods."""
    if name == "rule":
        return np.asarray(predict_rule_batch(X), dtype=int)
    return model.predict(X)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred):
    """Accuracy, macro precision/recall/F1, per-class scores, confusion matrix."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(LABELS)), zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": {LABELS[i]: float(per_class_f1[i]) for i in range(len(LABELS))},
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))),
    }


# --------------------------------------------------------------------------- #
# size
# --------------------------------------------------------------------------- #
def estimate_size(name, model):
    """Estimated raw parameter storage for inference, in float32 bytes.

    This is NOT a pickle size: it counts the numbers the C code has to store.
    """
    if name == "rule":
        params = 5  # the five thresholds in training/rules.py
        return {"parameters": params, "bytes": params * BYTES_PER_FLOAT,
                "detail": "5 thresholds"}

    if name == "logistic":
        clf = model.named_steps["clf"]
        scaler_params = model.named_steps["scaler"].mean_.size * 2
        params = clf.coef_.size + clf.intercept_.size + scaler_params
        return {"parameters": int(params), "bytes": int(params) * BYTES_PER_FLOAT,
                "detail": f"{clf.coef_.size} weights + {clf.intercept_.size} biases "
                          f"+ {scaler_params} scaler"}

    if name == "tree":
        tree = model.tree_
        internal = int(np.sum(tree.children_left != -1))
        leaves = int(tree.node_count - internal)
        # internal node: threshold (4 B) + feature id (1 B) + 2 children (2 B each)
        total = internal * (BYTES_PER_FLOAT + 1 + 4) + leaves * 1
        return {"parameters": int(tree.node_count), "bytes": int(total),
                "detail": f"{internal} internal nodes + {leaves} leaves"}

    if name == "mlp":
        clf = model.named_steps["clf"]
        params = sum(w.size + b.size for w, b in zip(clf.coefs_, clf.intercepts_))
        scaler_params = model.named_steps["scaler"].mean_.size * 2
        total = params + scaler_params
        return {"parameters": int(total), "bytes": int(total) * BYTES_PER_FLOAT,
                "detail": f"{params} weights/biases + {scaler_params} scaler"}

    raise ValueError(f"unknown method: {name}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    splits = load_data()
    X_train, y_train = splits["train"]
    models = train_models(X_train, y_train)

    for split in ("val", "test"):
        X, y = splits[split]
        print(f"\n=== {split} ===")
        print(f"{'method':10s} {'acc':>7s} {'macroF1':>9s}")
        for name in METHODS:
            metrics = evaluate(y, predict_ids(name, models[name], X))
            print(f"{name:10s} {metrics['accuracy']:7.4f} {metrics['macro_f1']:9.4f}")

    print("\n=== test size ===")
    for name in METHODS:
        size = estimate_size(name, models[name])
        print(f"{name:10s} {size['bytes']:6d} B  ({size['detail']})")


if __name__ == "__main__":
    main()
