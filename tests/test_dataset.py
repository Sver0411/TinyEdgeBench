"""Dataset tests: structure, reproducibility, and that the task is not trivial."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from dataset import generate
from dataset.generate import FEATURE_COLUMNS, LABELS

DATA_PATH = Path(__file__).resolve().parents[1] / "dataset" / "data.csv"


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(DATA_PATH)


def test_dataset_shape(data):
    assert len(data) == generate.N_ROWS
    assert list(data.columns) == ["seq_id", "split", "label"] + FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) <= 8


def test_no_missing_values(data):
    assert not data.isnull().any().any()
    assert np.isfinite(data[FEATURE_COLUMNS].to_numpy()).all()


def test_labels_are_balanced_and_valid(data):
    counts = data["label"].value_counts()
    assert set(counts.index) == set(LABELS)
    for label in LABELS:
        assert counts[label] == generate.N_ROWS // len(LABELS)


def test_splits_are_disjoint_sequences(data):
    by_split = {split: set(group["seq_id"]) for split, group in data.groupby("split")}
    assert set(by_split) == {"train", "val", "test"}
    assert not (by_split["train"] & by_split["val"])
    assert not (by_split["train"] & by_split["test"])
    assert not (by_split["val"] & by_split["test"])


def test_split_sizes_are_close_to_70_15_15(data):
    rows = data["split"].value_counts()
    total = len(data)
    for split, expected in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        assert abs(rows[split] / total - expected) < 0.02


def test_regeneration_is_reproducible():
    first = generate.build_dataset()
    second = generate.build_dataset()
    pd.testing.assert_frame_equal(first, second)


def test_committed_csv_matches_generator(data):
    pd.testing.assert_frame_equal(data, generate.build_dataset())


def test_classes_have_different_dynamics(data):
    means = data.groupby("label")["motion"].mean()
    assert means["NOISY"] > means["NORMAL"] * 2
    assert data.groupby("label")["drift"].mean()["NORMAL"] < 0.25


def test_no_single_feature_solves_the_task(data):
    """Guard against a dataset so simple that one feature separates everything."""
    train = data[data["split"] == "train"]
    test = data[data["split"] == "test"]
    best = 0.0
    for feature in FEATURE_COLUMNS:
        model = DecisionTreeClassifier(max_depth=3, random_state=0)
        model.fit(train[[feature]], train["label"])
        best = max(best, float((model.predict(test[[feature]]) == test["label"]).mean()))
    assert best < 0.90, f"single feature already reaches {best:.3f} accuracy"
