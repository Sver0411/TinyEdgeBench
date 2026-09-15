"""Export tests: the generated C code exists, is deterministic, and agrees
with the Python models on 100 samples."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dataset.generate import LABELS
from training import export_models, train

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE_MAIN = ROOT / "firmware" / "main"
MODELS_DIR = FIRMWARE_MAIN / "models"
N_PARITY_SAMPLES = 100

C_FILES = ("model_rule.c", "model_logistic.c", "model_tree.c", "model_mlp.c")


@pytest.fixture(scope="module")
def trained():
    splits = train.load_data()
    return splits, train.train_models(*splits["train"])


@pytest.fixture(scope="module")
def exported(trained):
    _, models = trained
    return export_models.export_all(models)


def test_export_writes_all_four_files(exported):
    assert len(exported) == 4
    for path in exported:
        assert path.exists()
        assert path.stat().st_size > 0


def test_export_is_deterministic(trained, exported):
    _, models = trained
    before = {path.name: path.read_text() for path in exported}
    again = export_models.export_all(models)
    for path in again:
        assert path.read_text() == before[path.name]


def test_generated_mlp_matches_architecture(exported):
    source = (MODELS_DIR / "model_mlp.c").read_text()
    h1, h2 = train.MLP_HIDDEN_LAYERS
    assert f"static const float w1[{h1}][{len(train.FEATURE_COLUMNS)}]" in source or \
           f"static const float w1[{h1}][8]" in source
    assert f"float h2[{h2}];" in source


def test_generated_tree_has_no_runtime_structures(exported):
    source = (MODELS_DIR / "model_tree.c").read_text()
    assert "if (features[" in source
    assert "for (" not in source


def test_rule_thresholds_in_c_match_python(exported):
    from training import rules
    source = (MODELS_DIR / "model_rule.c").read_text()
    for threshold in (rules.MOTION_NOISY, rules.DRIFT_FAST, rules.DRIFT_ACTIVE,
                      rules.DIRECTED_RATIO, rules.NOISY_RATIO):
        assert export_models.c_float(threshold) in source


def _compile_parity_binary(tmp_path):
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        pytest.skip("no C compiler available")

    binary = tmp_path / "parity_host"
    sources = [str(ROOT / "tests" / "parity_host.c")] + [str(MODELS_DIR / name) for name in C_FILES]
    result = subprocess.run(
        [compiler, "-std=c99", "-O2", "-Wall", f"-I{FIRMWARE_MAIN}", "-o", str(binary)] + sources + ["-lm"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def test_python_and_c_predictions_match(trained, exported, tmp_path):
    """Minimal parity check: 100 samples, Python prediction == C prediction."""
    from training.rules import predict_rule

    splits, models = trained
    X_test, _ = splits["test"]
    indices = np.random.default_rng(0).choice(len(X_test), N_PARITY_SAMPLES, replace=False)

    # float32 so Python and C work on exactly the same numbers
    samples = X_test[indices].astype(np.float32)

    sample_file = tmp_path / "samples.txt"
    sample_file.write_text(
        "\n".join(" ".join(f"{value:.9g}" for value in row) for row in samples) + "\n"
    )

    binary = _compile_parity_binary(tmp_path)
    output = subprocess.run([str(binary), str(sample_file)], capture_output=True, text=True, check=True)
    lines = [line.split() for line in output.stdout.strip().splitlines()]
    assert len(lines) == N_PARITY_SAMPLES

    for row, c_pred in zip(samples, lines):
        # the learned models are trained on class ids, so predict() returns ids
        python_pred = [
            predict_rule(row),
            int(models["logistic"].predict(row.reshape(1, -1))[0]),
            int(models["tree"].predict(row.reshape(1, -1))[0]),
            int(models["mlp"].predict(row.reshape(1, -1))[0]),
        ]
        assert [int(value) for value in c_pred] == python_pred


def test_parity_binary_reports_valid_classes(trained, exported, tmp_path):
    binary = _compile_parity_binary(tmp_path)
    sample_file = tmp_path / "one.txt"
    sample_file.write_text("24.0 50.0 300.0 0.0 0.0 0.0 1.0 0.14\n")
    output = subprocess.run([str(binary), str(sample_file)], capture_output=True, text=True, check=True)
    predictions = [int(value) for value in output.stdout.split()]
    assert len(predictions) == 4
    assert all(0 <= value < len(LABELS) for value in predictions)
