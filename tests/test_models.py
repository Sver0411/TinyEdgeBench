"""Model tests: every method runs, beats chance, and is sized consistently."""

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


def test_learned_methods_beat_the_hand_written_rules(scores):
    best_learned = max(scores[name]["accuracy"] for name in ("logistic", "tree", "mlp"))
    assert scores["rule"]["accuracy"] < best_learned


def test_learned_methods_agree_within_a_few_points(scores):
    values = [scores[name]["accuracy"] for name in ("logistic", "tree", "mlp")]
    assert max(values) - min(values) < 0.10


def test_rule_uses_no_training(trained):
    _, models = trained
    assert models["rule"] is None
    assert train.estimate_size("rule", None)["bytes"] == 20


def test_tree_depth_is_limited(trained):
    _, models = trained
    assert models["tree"].get_depth() <= train.TREE_MAX_DEPTH


def test_mlp_architecture_is_tiny(trained):
    _, models = trained
    clf = models["mlp"].named_steps["clf"]
    assert clf.hidden_layer_sizes == train.MLP_HIDDEN_LAYERS
    assert len(clf.hidden_layer_sizes) <= 2


def test_size_grows_with_complexity(trained):
    _, models = trained
    sizes = {name: train.estimate_size(name, models[name])["bytes"] for name in train.METHODS}
    assert sizes["rule"] < sizes["logistic"] < sizes["tree"] < sizes["mlp"]


def test_confusion_matrix_covers_every_test_row(trained, scores):
    splits, _ = trained
    _, y_test = splits["test"]
    for name in train.METHODS:
        assert scores[name]["confusion"].sum() == len(y_test)


def test_validation_and_test_scores_are_close(trained):
    splits, models = trained
    for name in ("logistic", "tree", "mlp"):
        val = train.evaluate(splits["val"][1], train.predict_ids(name, models[name], splits["val"][0]))
        test = train.evaluate(splits["test"][1], train.predict_ids(name, models[name], splits["test"][0]))
        assert abs(val["accuracy"] - test["accuracy"]) < 0.05
