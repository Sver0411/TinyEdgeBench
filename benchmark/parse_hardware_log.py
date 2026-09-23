"""Validate machine-readable ESP32 logs and write traceable result files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "hardware"
MODELS = ("rule", "logistic", "tree", "mlp")
FIELDS = ("model", "run", "accuracy", "macro_f1", "correct", "samples",
          "checksum", "latency_min_us", "latency_mean_us", "latency_p50_us",
          "latency_p95_us", "latency_p99_us", "latency_max_us",
          "heap_before_bytes", "heap_after_bytes", "heap_minimum_bytes",
          "main_task_stack_configured_bytes", "main_task_stack_minimum_remaining_bytes",
          "flash_delta_bytes", "static_ram_delta_bytes", "git_commit", "git_dirty")
BUILD_MISMATCH = "ERROR: build-size result does not belong to this hardware benchmark."


def parse_log(text: str):
    records = []
    current = None
    for line in text.splitlines():
        line = line.strip()
        for marker in ("TEB_BEGIN", "TEB_RESULT", "TEB_END", "TEB_ERROR"):
            at = line.find(marker + " ")
            if at < 0:
                continue
            data = json.loads(line[at + len(marker) + 1:])
            if marker == "TEB_ERROR":
                raise ValueError(f"hardware correctness failed: {data}")
            if marker == "TEB_BEGIN":
                if current is not None:
                    raise ValueError("nested benchmark begin")
                current = {"metadata": data}
            elif current is None:
                raise ValueError("result/end without begin")
            elif marker == "TEB_RESULT":
                if "result" in current:
                    raise ValueError("duplicate result")
                current["result"] = data
            else:
                if "result" not in current:
                    raise ValueError("end without result")
                if len({current["metadata"]["model"], current["result"]["model"], data["model"]}) != 1:
                    raise ValueError("model mismatch across markers")
                validate(current)
                records.append(current)
                current = None
            break
    if current is not None or not records:
        raise ValueError("incomplete or missing benchmark record")
    return records


def validate(record):
    meta, result = record["metadata"], record["result"]
    required = ("git_commit", "source_sha256", "model_sha256",
                "sdkconfig_sha256", "idf", "compiler", "cpu_mhz",
                "optimization", "image_flash_size", "subset_sha256")
    if any(not meta.get(key) for key in required):
        raise ValueError("missing hardware provenance")
    if not isinstance(meta.get("git_dirty"), bool):
        raise ValueError("missing git dirty provenance")
    if meta["model"] not in MODELS or meta["chip"] != "ESP32-S3":
        raise ValueError("unexpected model/chip")
    if meta["samples"] != result["samples"] or meta["samples"] != 256:
        raise ValueError("unexpected sample count")
    if (meta["timed_batches"] < 2000 or meta["inferences_per_batch"] != 64
            or meta["iterations"] != meta["timed_batches"] * meta["inferences_per_batch"]
            or meta["warmup"] < 100):
        raise ValueError("invalid batch timing configuration")
    if meta["dataset_sha256"] != hashlib.sha256((ROOT / "dataset/data.csv").read_bytes()).hexdigest():
        raise ValueError("dataset hash mismatch")
    match = re.search(r'const char benchmark_subset_sha256\[\] = "([0-9a-f]{64})"',
                      (ROOT / "firmware/main/benchmark_dataset.c").read_text())
    if not match or meta["subset_sha256"] != match.group(1):
        raise ValueError("benchmark subset hash mismatch")
    cm = result["confusion"]
    if len(cm) != 16 or sum(cm) != result["samples"]:
        raise ValueError("invalid confusion matrix")
    if sum(cm[i * 4 + i] for i in range(4)) != result["correct"]:
        raise ValueError("correct count mismatch")
    latency = [result[f"latency_{p}_us"] for p in ("min", "p50", "p95", "p99", "max")]
    if not all(isinstance(value, (float, int)) and math.isfinite(value) and value >= 0
               for value in latency):
        raise ValueError("invalid latency")
    if latency != sorted(latency):
        raise ValueError("latency percentiles out of order")
    if result["heap_after_bytes"] > result["heap_before_bytes"] + 4096:
        raise ValueError("unexpected heap increase")


def macro_f1(confusion):
    matrix = [confusion[i:i+4] for i in range(0, 16, 4)]
    f1 = []
    for cls in range(4):
        tp = matrix[cls][cls]
        false_positive = sum(matrix[row][cls] for row in range(4)) - tp
        false_negative = sum(matrix[cls]) - tp
        denominator = 2 * tp + false_positive + false_negative
        f1.append(2 * tp / denominator if denominator else 0.0)
    return sum(f1) / 4


def load_build_sizes(path=ROOT / "results/flash_size.csv"):
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["method"]: row for row in csv.DictReader(handle)}


def write_results(log_paths, output=RESULTS, build_sizes=ROOT / "results/flash_size.csv"):
    sizes = load_build_sizes(Path(build_sizes))
    rows, metadata = [], []
    captures = []
    run_counts = {model: 0 for model in MODELS}
    for path in map(Path, log_paths):
        content = path.read_text(errors="replace")
        for record in parse_log(content):
            meta, result = record["metadata"], record["result"]
            model = meta["model"]
            run_counts[model] += 1
            run = run_counts[model]
            row = {key: result[key] for key in FIELDS if key in result}
            row.update(model=model, run=run,
                       accuracy=result["correct"] / result["samples"],
                       macro_f1=macro_f1(result["confusion"]),
                       git_commit=meta["git_commit"], git_dirty=meta["git_dirty"])
            size = sizes.get(model)
            if size is None or any(str(size.get(key, "")).lower() != str(meta[key]).lower()
                                   for key in ("git_commit", "git_dirty", "source_sha256",
                                               "sdkconfig_sha256", "model_sha256")):
                raise ValueError(BUILD_MISMATCH)
            row["flash_delta_bytes"] = size["flash_delta_bytes"]
            row["static_ram_delta_bytes"] = size["static_ram_delta_bytes"]
            rows.append(row)
            metadata.append({**meta, "run": run, "raw_log": path.name})
        captures.append((path.name, content))
    for key in ("git_commit", "git_dirty", "source_sha256", "sdkconfig_sha256", "dataset_sha256", "subset_sha256",
                "idf", "compiler", "cpu_mhz", "optimization", "image_flash_size"):
        if len({item[key] for item in metadata}) != 1:
            raise ValueError(f"mixed hardware configurations: {key}")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw_serial_logs"
    raw.mkdir(exist_ok=True)
    for old in raw.glob("*.log"):
        old.unlink()
    for name, content in captures:
        # Serial boot banners may contain a MAC address; retain only benchmark records.
        public_lines = [line for line in content.splitlines()
                        if any(marker + " " in line for marker in
                               ("TEB_BEGIN", "TEB_RESULT", "TEB_END", "TEB_ERROR"))]
        (raw / name).write_text("\n".join(public_lines) + "\n")
    with (output / "hardware_benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / "hardware_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument("--build-sizes", type=Path, default=ROOT / "results/flash_size.csv")
    args = parser.parse_args()
    rows = write_results(args.logs, args.output, args.build_sizes)
    print(f"wrote {len(rows)} validated runs to {args.output}")


if __name__ == "__main__":
    main()
