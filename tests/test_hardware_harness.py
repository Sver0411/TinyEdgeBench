"""Host checks for the fixed ESP32 dataset and machine-readable log contract."""

import hashlib
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from benchmark import export_dataset, parse_hardware_log as parser

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "firmware/main"


def test_export_is_current_and_has_256_test_rows():
    assert export_dataset.OUT.read_text() == export_dataset.render()
    rows = export_dataset.selected_rows()
    assert len(rows) == 256
    assert rows.label.value_counts().to_dict() == dict.fromkeys(
        ("NORMAL", "RAPID_CHANGE", "SLOW_DRIFT", "NOISY"), 64)
    assert rows.groupby("label").seq_id.nunique().to_dict() == dict.fromkeys(
        ("NORMAL", "RAPID_CHANGE", "SLOW_DRIFT", "NOISY"), 64)
    source = export_dataset.OUT.read_text()
    assert source.count("    {") == 256
    assert hashlib.sha256(export_dataset.DATA.read_bytes()).hexdigest() in source


def test_every_exported_c_prediction_matches_oracle(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    harness = tmp_path / "check.c"
    harness.write_text('''
#include <stdio.h>
#include "benchmark_dataset.h"
int main(void) {
    int (*predict[])(const float *) = {
        predict_rule, predict_logistic, predict_tree, predict_mlp
    };
    const uint8_t *expected[] = {
        benchmark_expected_rule, benchmark_expected_logistic,
        benchmark_expected_tree, benchmark_expected_mlp
    };
    for (int m = 0; m < 4; m++) for (int i = 0; i < BENCHMARK_SAMPLES; i++)
        if (predict[m](benchmark_features[i]) != expected[m][i]) return m + 1;
    return 0;
}
''')
    sources = [harness, MAIN / "benchmark_dataset.c",
               *(MAIN / "models" / f"model_{m}.c" for m in parser.MODELS)]
    binary = tmp_path / "check"
    subprocess.run([cc, "-O2", "-I", str(MAIN), "-I", str(MAIN / "models"),
                    *map(str, sources), "-lm", "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def sample_log(model="rule"):
    sha = hashlib.sha256((ROOT / "dataset/data.csv").read_bytes()).hexdigest()
    subset_sha = re.search(r'const char benchmark_subset_sha256\[\] = "([0-9a-f]{64})"',
                           (MAIN / "benchmark_dataset.c").read_text()).group(1)
    meta = {"model": model, "chip": "ESP32-S3", "samples": 256,
            "warmup": 100, "iterations": 128000, "timed_batches": 2000,
            "inferences_per_batch": 64, "dataset_sha256": sha,
            "subset_sha256": subset_sha,
            "git_commit": "test", "git_dirty": False, "source_sha256": "source",
            "model_sha256": "model", "sdkconfig_sha256": "config",
            "cpu_mhz": 240, "idf": "v5.4.4", "compiler": "GCC 14.2.0",
            "optimization": "performance", "image_flash_size": "16MB"}
    result = {"model": model, "samples": 256, "correct": 256,
              "confusion": [64 if i % 5 == 0 else 0 for i in range(16)],
              "latency_min_us": 0.125, "latency_p50_us": 1.125,
              "latency_p95_us": 2.125, "latency_p99_us": 3.125,
              "latency_max_us": 4.125, "heap_before_bytes": 1000,
              "heap_after_bytes": 1000}
    return "\n".join(f"{key} {json.dumps(value)}" for key, value in
                     (("TEB_BEGIN", meta), ("TEB_RESULT", result),
                      ("TEB_END", {"model": model})))


def test_parser_and_schema(tmp_path):
    log = tmp_path / "sample.log"
    log.write_text(sample_log())
    size = tmp_path / "sizes.csv"
    with size.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "git_commit", "git_dirty",
            "source_sha256", "model_sha256", "sdkconfig_sha256",
            "flash_delta_bytes", "static_ram_delta_bytes"))
        writer.writeheader()
        writer.writerow(dict(method="rule", git_commit="test", git_dirty="false",
                             source_sha256="source", model_sha256="model",
                             sdkconfig_sha256="config", flash_delta_bytes=40,
                             static_ram_delta_bytes=0))
    rows = parser.write_results([log], tmp_path / "out", size)
    assert len(rows) == 1 and rows[0]["accuracy"] == 1
    assert rows[0]["macro_f1"] == 1
    assert rows[0]["latency_p50_us"] == 1.125
    with (tmp_path / "out/hardware_benchmark.csv").open() as handle:
        assert handle.readline().strip().split(",") == list(parser.FIELDS)
    metadata = json.loads((tmp_path / "out/hardware_metadata.json").read_text())
    assert metadata[0]["dataset_sha256"] == hashlib.sha256(
        (ROOT / "dataset/data.csv").read_bytes()).hexdigest()
    log.write_text(sample_log().replace('"git_commit": "test"',
                                        '"git_commit": "other"'))
    with pytest.raises(ValueError, match="build-size result does not belong"):
        parser.write_results([log], tmp_path / "bad", size)


@pytest.mark.parametrize("bad", ["", "TEB_RESULT {}", "TEB_BEGIN {}",
                                   sample_log().replace("TEB_END", "TEB_ERROR"),
                                   sample_log().replace('"correct": 256', '"correct": 0')])
def test_invalid_log_is_rejected(bad):
    with pytest.raises((ValueError, KeyError)):
        parser.parse_log(bad)


def test_model_selection_is_single_source():
    cmake = (MAIN / "CMakeLists.txt").read_text()
    assert 'set(TINYEDGEBENCH_MODEL "rule"' in cmake
    assert 'STREQUAL "all"' not in cmake
    assert 'list(FILTER TEB_SOURCE_FILES EXCLUDE REGEX' in cmake
    for model in parser.MODELS:
        branch = re.search(rf'STREQUAL "{model}"\)(.*?)(?:elseif|else\()', cmake, re.S)
        assert branch is not None
        assert branch.group(1).count('"models/model_') == 1
        assert f'"models/model_{model}.c"' in branch.group(1)


def test_nearest_rank_percentile(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler unavailable")
    source = tmp_path / "percentile.c"
    source.write_text('''
#include "benchmark_stats.h"
int main(void) {
    uint32_t values[100];
    for (int i = 0; i < 100; i++) values[i] = i + 1;
    return teb_percentile(values, 100, 50, 100) != 50 ||
           teb_percentile(values, 100, 95, 100) != 95 ||
           teb_percentile(values, 100, 99, 100) != 99;
}
''')
    binary = tmp_path / "percentile"
    subprocess.run([cc, "-I", str(MAIN), str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_committed_hardware_results_have_one_clean_build_provenance():
    output = ROOT / "results/hardware"
    metadata = json.loads((output / "hardware_metadata.json").read_text())
    with (output / "hardware_benchmark.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (ROOT / "results/flash_size.csv").open(newline="") as handle:
        sizes = {row["method"]: row for row in csv.DictReader(handle)}
    assert len(rows) == len(metadata) == 12
    assert {model: sum(row["model"] == model for row in rows)
            for model in parser.MODELS} == dict.fromkeys(parser.MODELS, 3)
    assert all(item["git_dirty"] is False for item in metadata)
    for key in ("git_commit", "dataset_sha256", "subset_sha256", "sdkconfig_sha256",
                "cpu_mhz", "idf", "optimization", "source_sha256"):
        assert len({item[key] for item in metadata}) == 1
    for item in metadata:
        size = sizes[item["model"]]
        for key in ("git_commit", "source_sha256", "sdkconfig_sha256", "model_sha256"):
            assert item[key] == size[key]
        assert size["git_dirty"].lower() == "false"
        capture = (output / "raw_serial_logs" / item["raw_log"]).read_text()
        assert "MAC:" not in capture and "Serial port:" not in capture
        records = parser.parse_log(capture)
        assert len(records) == 1
        assert records[0]["metadata"]["git_commit"] == item["git_commit"]
        assert records[0]["metadata"]["model"] == item["model"]
