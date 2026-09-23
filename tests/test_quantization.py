"""Numerical contract for portable scalar INT8 and committed v0.3 evidence."""

import csv
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pytest

from benchmark.export_dataset import selected_rows
from benchmark import v03_parse_hardware as hardware
from dataset.generate import FEATURE_COLUMNS
from training import quantize, train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def params():
    return json.loads((ROOT / "results/v0.3/quantization_params.json").read_text())


@pytest.fixture(scope="module")
def host_lib(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    dest = tmp_path_factory.mktemp("int8") / "models.so"
    model_dir = ROOT / "firmware/main/models"
    subprocess.run([cc, "-shared", "-fPIC", "-O2", "-DTEB_HOST_DEBUG",
                    "-I", str(model_dir), str(model_dir / "model_logistic_int8.c"),
                    str(model_dir / "model_mlp_int8.c"), "-o", str(dest)], check=True)
    return ctypes.CDLL(str(dest))


class LogisticDebug(ctypes.Structure):
    _fields_ = [("input", ctypes.c_int8 * 8), ("logits", ctypes.c_int32 * 4),
                ("prediction", ctypes.c_int)]


class MlpDebug(ctypes.Structure):
    _fields_ = [("input", ctypes.c_int8 * 8), ("accumulator1", ctypes.c_int32 * 8),
                ("activation1", ctypes.c_int8 * 8), ("accumulator2", ctypes.c_int32 * 8),
                ("activation2", ctypes.c_int8 * 8), ("logits", ctypes.c_int32 * 4),
                ("prediction", ctypes.c_int)]


def test_export_and_quantization_params_are_current():
    for path, expected in quantize.generate_artifacts().items():
        assert path.read_text() == expected, path


def test_train_only_calibration_and_split_stability(params):
    splits = train.load_data()
    x_train, _ = splits["train"]
    assert params["calibration_split"] == "train"
    assert params["selection_split"] == "none_fixed_before_test"
    assert params["test_used_for_calibration"] is False
    assert len(splits["test"][0]) == 3000
    for family in ("logistic", "mlp"):
        scaled = quantize.standardized(x_train, params[family])
        assert params[family]["input_calibration_max_abs"] == float(np.max(np.abs(scaled)))
        assert params[family]["input_scale"] == quantize.scale(scaled)
    with (ROOT / "results/benchmark.csv").open(newline="") as handle:
        baseline = {row["method"]: row for row in csv.DictReader(handle)}
    with (ROOT / "results/v0.3/offline_quantization.csv").open(newline="") as handle:
        offline = {row["variant"]: row for row in csv.DictReader(handle)}
    for family in ("logistic", "mlp"):
        assert round(float(offline[f"{family}_fp32"]["accuracy"]), 4) == float(baseline[family]["accuracy"])
        assert round(float(offline[f"{family}_fp32"]["macro_f1"]), 4) == float(baseline[family]["macro_f1"])


def test_int8_ranges_bias_and_overflow_bounds(params):
    for family in ("logistic", "mlp"):
        layers = params[family].get("layers", [params[family].get("layer")])
        for layer in layers:
            weights = np.asarray(layer["weights"])
            bias = np.asarray(layer["bias"])
            assert weights.min() >= -127 and weights.max() <= 127
            assert bias.min() >= -(1 << 31) and bias.max() < (1 << 31)
            assert layer["accumulator_absolute_bound"] < (1 << 31)
            assert layer["accumulator_absolute_bound"] >= max(abs(bias)) + weights.shape[1] * 127 * 127
            if "multiplier" in layer:
                assert layer["shift"] == 24
                assert layer["requantization_product_bound"] + (1 << 23) < (1 << 63)


def test_rounding_saturation_and_fixed_point_python_and_c(tmp_path):
    assert quantize.round_half_away([.5, -.5, 1.5, -1.5]).tolist() == [1, -1, 2, -2]
    assert quantize.quantize([1000, -1000, .5, -.5], 1).tolist() == [127, -127, 1, -1]
    values, _, _ = quantize.requantize_relu(np.array([-1000, 0, 1, 253, 100000]), 1 << 23)
    assert values.tolist() == [0, 0, 1, 127, 127]
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    source = tmp_path / "arithmetic.c"
    source.write_text('''
#include "quantized_ops.h"
int main(void) {
 return teb_round_half_away_float(0.5f) != 1 ||
        teb_round_half_away_float(-0.5f) != -1 ||
        teb_round_half_away_float(1.5f) != 2 ||
        teb_round_half_away_float(-1.5f) != -2 ||
        teb_quantize_symmetric(1000.0f) != 127 ||
        teb_quantize_symmetric(-1000.0f) != -127 ||
        teb_requantize_relu(1, 1 << 23, 24) != 1 ||
        teb_requantize_relu(-1000, 1 << 23, 24) != 0 ||
        teb_requantize_relu(100000, 1 << 23, 24) != 127;
}
''')
    binary = tmp_path / "arithmetic"
    subprocess.run([cc, "-I", str(ROOT / "firmware/main/models"),
                    str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


@pytest.mark.parametrize("family,debug_cls", [("logistic", LogisticDebug), ("mlp", MlpDebug)])
def test_python_host_c_int8_parity_and_intermediates(params, host_lib, family, debug_cls):
    x = np.ascontiguousarray(selected_rows()[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    expected, states, _ = quantize.infer(x, params[family], family)
    predict = getattr(host_lib, f"predict_{family}_int8")
    predict.argtypes = [ctypes.POINTER(ctypes.c_float)]
    predict.restype = ctypes.c_int
    actual = [predict(row.ctypes.data_as(ctypes.POINTER(ctypes.c_float))) for row in x]
    assert actual == expected.tolist()
    debug = getattr(host_lib, f"teb_debug_{family}_int8")
    debug.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(debug_cls)]
    for i in range(10):
        state = debug_cls()
        debug(x[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(state))
        assert state.prediction == int(expected[i])
        for field in ("input", "logits") + (("accumulator1", "activation1",
                                               "accumulator2", "activation2") if family == "mlp" else ()):
            assert list(getattr(state, field)) == states[field][i].tolist(), (family, field, i)


def test_quantization_hash_is_embedded_by_build():
    source = (ROOT / "firmware/main/CMakeLists.txt").read_text()
    assert 'quantization_params.json" TEB_QUANT_HASH' in source
    digest = hashlib.sha256((ROOT / "results/v0.3/quantization_params.json").read_bytes()).hexdigest()
    assert len(digest) == 64


def test_v03_parser_rejects_stale_build_size_and_quantization(tmp_path):
    quant_hash = hashlib.sha256((ROOT / "results/v0.3/quantization_params.json").read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256((ROOT / "dataset/data.csv").read_bytes()).hexdigest()
    subset_hash = re.search(r'benchmark_subset_sha256\[\] = "([0-9a-f]{64})"',
                            (ROOT / "firmware/main/benchmark_dataset.c").read_text()).group(1)
    paths = []
    for model in hardware.ALL:
        family = "baseline" if model == "timing_baseline" else model.split("_")[0]
        precision = "int8" if model.endswith("_int8") else "fp32" if model.endswith("_fp32") else "native"
        meta = dict(model=model, model_family=family, precision=precision,
                    implementation="portable_scalar", chip="ESP32-S3", samples=256,
                    warmup=100, iterations=128000, timed_batches=2000, inferences_per_batch=64,
                    git_commit="a" * 40, git_dirty=False, source_sha256="b" * 64,
                    model_sha256="none" if model == "timing_baseline" else "c" * 64,
                    sdkconfig_sha256="d" * 64, quantization_sha256=quant_hash,
                    dataset_sha256=dataset_hash, subset_sha256=subset_hash,
                    cpu_mhz=160, idf="v5.4.4", compiler="GCC 14.2.0",
                    optimization="performance", image_flash_size="16MB",
                    power_management="disabled")
        result = dict(model=model, samples=256, correct=256,
                      confusion=[64 if i % 5 == 0 else 0 for i in range(16)],
                      heap_before_bytes=1000, heap_after_bytes=1000,
                      latency_min_us=.125, latency_mean_us=.25, latency_p50_us=.25,
                      latency_p95_us=.375, latency_p99_us=.5, latency_max_us=.625)
        for run in range(1, 4):
            path = tmp_path / f"{model}_{run}.log"
            path.write_text("\n".join(f"{marker} {json.dumps(data)}" for marker, data in
                                      (("TEB_BEGIN", meta), ("TEB_RESULT", result),
                                       ("TEB_END", {"model": model}))))
            paths.append(path)
    flash = tmp_path / "flash.csv"
    fields = ("variant", "git_commit", "git_dirty", "source_sha256", "model_sha256",
              "sdkconfig_sha256", "quantization_sha256", "raw_parameter_bytes",
              "declared_local_scratch_bytes", "flash_delta_bytes", "static_ram_delta_bytes")
    with flash.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for model in hardware.VARIANTS:
            writer.writerow(dict(variant=model, git_commit="a" * 40, git_dirty="False",
                                 source_sha256="b" * 64, model_sha256="c" * 64,
                                 sdkconfig_sha256="d" * 64, quantization_sha256=quant_hash,
                                 raw_parameter_bytes="12", declared_local_scratch_bytes="4",
                                 flash_delta_bytes="20", static_ram_delta_bytes="0"))
    rows = hardware.write_results(paths, flash, tmp_path / "parsed")
    assert len(rows) == 18
    assert (tmp_path / "parsed/baseline_timing.csv").exists()
    stale = flash.read_text().replace("c" * 64, "e" * 64, 1)
    flash.write_text(stale)
    with pytest.raises(ValueError, match="build-size result does not belong"):
        hardware.write_results(paths, flash, tmp_path / "rejected")
    assert not (tmp_path / "rejected/hardware_benchmark.csv").exists()
    bad_quant = paths[0].read_text().replace(quant_hash, "f" * 64)
    paths[0].write_text(bad_quant)
    with pytest.raises(ValueError, match="quantization config hash mismatch"):
        hardware.write_results(paths, flash, tmp_path / "wrong-quant")


def test_v03_committed_result_schema():
    output = ROOT / "results/v0.3"
    if not (output / "hardware_benchmark.csv").exists():
        pytest.skip("hardware run pending")
    with (output / "hardware_benchmark.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads((output / "hardware_metadata.json").read_text())
    with (output / "flash_size.csv").open(newline="") as handle:
        sizes = {row["variant"]: row for row in csv.DictReader(handle)}
    assert len(rows) == 18 and len(metadata) == 21
    assert {row["variant"] for row in rows} == set(hardware.VARIANTS)
    assert all(sum(row["variant"] == variant for row in rows) == 3 for variant in hardware.VARIANTS)
    assert all(item["git_dirty"] is False for item in metadata)
    for key in ("git_commit", "source_sha256", "sdkconfig_sha256", "dataset_sha256",
                "subset_sha256", "quantization_sha256", "cpu_mhz", "idf", "optimization"):
        assert len({item[key] for item in metadata}) == 1
    for row in rows:
        size = sizes[row["variant"]]
        for key in ("git_commit", "source_sha256", "sdkconfig_sha256",
                    "model_sha256", "quantization_sha256"):
            assert row[key] == size[key]
    for item in metadata:
        text = (output / "raw_serial_logs" / item["raw_log"]).read_text()
        assert "MAC:" not in text and "Serial port:" not in text
        assert hardware.parse_log(text)[0]["model"] == item["model"]
