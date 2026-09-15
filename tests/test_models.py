"""Model tests: every method runs, produces valid output, and stays within
the documented structural limits. No test pins down experimental rankings -
those are results, not correctness."""

import numpy as np
import pytest

from dataset.generate import LABELS
from training import train

RANDOM_BASELINE = 1.0 / len(LABELS)


@pytest.fixture(scope="module")
def trained():
    splits = train.load_data()
    models = train.train_models(*splits["train"])
    return splits, models


@pytest.fixture(scope="module")
def scores(trained):
    splits, models = trained
    X_test, y_test = splits["test"]
    return {name: train.evaluate(y_test, train.predict_ids(name, models[name], X_test))
            for name in train.METHODS}


def test_all_methods_predict_valid_class_ids(trained):
    splits, models = trained
    X_test, _ = splits["test"]
    for name in train.METHODS:
        ids = train.predict_ids(name, models[name], X_test)
        assert len(ids) == len(X_test)
        assert set(np.unique(ids)).issubset(set(range(len(LABELS))))


def test_all_methods_beat_random_baseline(scores):
    for name, metrics in scores.items():
        assert metrics["accuracy"] > RANDOM_BASELINE + 0.20, name


def test_rule_uses_no_training(trained):
    _, models = trained
    assert models["rule"] is None
    assert train.estimate_size("rule", None)["raw_constants_bytes"] == 5 * 4  # 5 thresholds


def test_tree_depth_is_limited(trained):
    _, models = trained
    assert models["tree"].get_depth() <= train.TREE_MAX_DEPTH


def test_mlp_architecture_is_tiny(trained):
    _, models = trained
    clf = models["mlp"].named_steps["clf"]
    assert clf.hidden_layer_sizes == train.MLP_HIDDEN_LAYERS
    assert len(clf.hidden_layer_sizes) <= 2


def test_size_estimate_matches_stored_parameters(trained):
    """Raw-constant counts must match what each model actually stores."""
    _, models = trained
    scaler_constants = len(train.FEATURE_COLUMNS) * 2  # mean + scale per feature

    logreg = models["logistic"].named_steps["clf"]
    assert train.estimate_size("logistic", models["logistic"])["raw_constants_bytes"] == (
        logreg.coef_.size + logreg.intercept_.size + scaler_constants) * 4

    mlp = models["mlp"].named_steps["clf"]
    expected = sum(w.size + b.size for w, b in zip(mlp.coefs_, mlp.intercepts_)) + scaler_constants
    assert train.estimate_size("mlp", models["mlp"])["raw_constants_bytes"] == expected * 4

    # the tree is exported as nested comparisons: no raw constants at all
    tree_size = train.estimate_size("tree", models["tree"])
    assert tree_size["raw_constants_bytes"] is None
    assert tree_size["structural_estimate_bytes"] > 0


def test_confusion_matrix_covers_every_test_row(trained, scores):
    splits, _ = trained
    _, y_test = splits["test"]
    for name in train.METHODS:
        assert scores[name]["confusion"].sum() == len(y_test)


