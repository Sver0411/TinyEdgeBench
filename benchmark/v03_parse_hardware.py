"""Validate v0.3 serial evidence and pair each run with its exact ELF build."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/v0.3"
VARIANTS = ("rule", "tree", "logistic_fp32", "logistic_int8", "mlp_fp32", "mlp_int8")
ALL = ("timing_baseline", *VARIANTS)
BUILD_MISMATCH = "ERROR: build-size result does not belong to this hardware benchmark."
FIELDS = ("variant", "model_family", "precision", "implementation", "run", "git_commit",
          "git_dirty", "source_sha256", "model_sha256", "sdkconfig_sha256",
          "dataset_sha256", "subset_sha256", "quantization_sha256", "cpu_mhz", "idf",
          "optimization", "samples", "correct", "accuracy", "macro_f1",
          "raw_parameter_bytes", "declared_local_scratch_bytes", "flash_delta_bytes",
          "static_ram_delta_bytes", "inference_heap_alloc_bytes",
          "latency_raw_mean_us", "latency_raw_p50_us", "latency_raw_p95_us",
          "latency_raw_p99_us", "baseline_latency_us", "baseline_latency_p50_us",
          "latency_adjusted_mean_us", "latency_adjusted_p50_us",
          "latency_adjusted_p95_us", "latency_adjusted_p99_us")


def parse_log(content):
    found = {}
    for line in content.splitlines():
        for marker in ("TEB_BEGIN", "TEB_RESULT", "TEB_END", "TEB_ERROR"):
            at = line.find(marker + " ")
            if at >= 0:
                if marker == "TEB_ERROR":
                    raise ValueError(f"hardware prediction parity failed: {line[at:]}")
                if marker in found:
                    raise ValueError(f"duplicate {marker}")
                found[marker] = json.loads(line[at + len(marker) + 1:])
                break
    if set(found) != {"TEB_BEGIN", "TEB_RESULT", "TEB_END"}:
        raise ValueError("incomplete hardware capture")
    meta, result, end = (found[key] for key in ("TEB_BEGIN", "TEB_RESULT", "TEB_END"))
    if len({meta["model"], result["model"], end["model"]}) != 1:
        raise ValueError("mixed model markers")
    return meta, result


def f1(confusion):
    values = []
    for cls in range(4):
        tp = confusion[cls * 4 + cls]
        fp = sum(confusion[row * 4 + cls] for row in range(4)) - tp
        fn = sum(confusion[cls * 4:cls * 4 + 4]) - tp
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return statistics.mean(values)


def validate(meta, result):
    model = meta["model"]
    if model not in ALL or meta["chip"] != "ESP32-S3":
        raise ValueError("unexpected model or chip")
    if meta["implementation"] != "portable_scalar":
        raise ValueError("unexpected implementation")
    family = model.split("_")[0] if model != "timing_baseline" else "baseline"
    precision = "int8" if model.endswith("_int8") else "fp32" if model.endswith("_fp32") else "native"
    if (meta["model_family"], meta["precision"]) != (family, precision):
        raise ValueError("incorrect variant identity")
    if (meta["samples"] != 256 or result["samples"] != 256 or meta["warmup"] != 100
            or meta["timed_batches"] != 2000 or meta["inferences_per_batch"] != 64
            or meta["iterations"] != 128000):
        raise ValueError("hardware timing protocol differs")
    if not isinstance(meta["git_dirty"], bool) or len(meta["git_commit"]) != 40:
        raise ValueError("invalid Git provenance")
    for key in ("source_sha256", "model_sha256", "sdkconfig_sha256",
                "dataset_sha256", "subset_sha256", "quantization_sha256"):
        if key != "model_sha256" or model not in ("timing_baseline",):
            if not re.fullmatch(r"[0-9a-f]{64}", meta[key]):
                raise ValueError(f"invalid {key}")
    if meta["dataset_sha256"] != hashlib.sha256((ROOT / "dataset/data.csv").read_bytes()).hexdigest():
        raise ValueError("dataset hash mismatch")
    subset = re.search(r'const char benchmark_subset_sha256\[\] = "([0-9a-f]{64})"',
                       (ROOT / "firmware/main/benchmark_dataset.c").read_text())
    if not subset or meta["subset_sha256"] != subset.group(1):
        raise ValueError("subset hash mismatch")
    if meta["quantization_sha256"] != hashlib.sha256(
            (ROOT / "results/v0.3/quantization_params.json").read_bytes()).hexdigest():
        raise ValueError("quantization config hash mismatch")
    latency = [result[f"latency_{key}_us"] for key in ("min", "mean", "p50", "p95", "p99", "max")]
    if not all(isinstance(x, (float, int)) and math.isfinite(x) and x >= 0 for x in latency):
        raise ValueError("invalid latency")
    if [latency[i] for i in (0, 2, 3, 4, 5)] != sorted(latency[i] for i in (0, 2, 3, 4, 5)):
        raise ValueError("unordered latency percentiles")
    if model != "timing_baseline":
        cm = result["confusion"]
        if len(cm) != 16 or sum(cm) != 256 or sum(cm[i * 5] for i in range(4)) != result["correct"]:
            raise ValueError("invalid confusion matrix")
    return model


def write_results(log_paths, flash_csv, output=OUTPUT, *, allow_dirty=False):
    output = Path(output)
    if allow_dirty and (output.resolve() == OUTPUT.resolve()
                        or OUTPUT.resolve() in output.resolve().parents):
        raise ValueError("dirty runs cannot overwrite formal v0.3 results")
    with Path(flash_csv).open(newline="") as handle:
        sizes = {row["variant"]: row for row in csv.DictReader(handle)}
    records, captures = [], []
    counts = {model: 0 for model in ALL}
    for path in map(Path, log_paths):
        content = path.read_text(errors="replace")
        meta, result = parse_log(content)
        model = validate(meta, result)
        if meta["git_dirty"] and not allow_dirty:
            raise ValueError("formal results require git_dirty=false")
        counts[model] += 1
        records.append((meta, result, counts[model], path.name))
        captures.append((path.name, content))
    if any(counts[model] < 3 for model in ALL) or len(records) != 21:
        raise ValueError("need exactly three baseline and six-model runs")
    for key in ("git_commit", "git_dirty", "source_sha256", "sdkconfig_sha256",
                "dataset_sha256", "subset_sha256", "quantization_sha256", "cpu_mhz",
                "idf", "compiler", "optimization", "image_flash_size", "power_management"):
        if len({meta[key] for meta, _, _, _ in records}) != 1:
            raise ValueError(f"mixed hardware configuration: {key}")
    for meta, _, _, _ in records:
        if meta["model"] == "timing_baseline":
            continue
        size = sizes.get(meta["model"])
        if size is None or any(str(size.get(key, "")).lower() != str(meta[key]).lower()
                               for key in ("git_commit", "git_dirty", "source_sha256",
                                           "model_sha256", "sdkconfig_sha256", "quantization_sha256")):
            raise ValueError(BUILD_MISMATCH)
    baselines = {run: result for meta, result, run, _ in records
                 if meta["model"] == "timing_baseline"}
    rows = []
    for meta, result, run, _ in records:
        variant = meta["model"]
        if variant == "timing_baseline":
            continue
        size = sizes[variant]
        base = baselines[run]
        row = {"variant": variant, "model_family": meta["model_family"],
               "precision": meta["precision"], "implementation": meta["implementation"],
               "run": run,
               **{key: meta[key] for key in ("git_commit", "git_dirty", "source_sha256",
                    "model_sha256", "sdkconfig_sha256", "dataset_sha256", "subset_sha256",
                    "quantization_sha256", "cpu_mhz", "idf", "optimization")},
               "samples": result["samples"], "correct": result["correct"],
               "accuracy": result["correct"] / 256, "macro_f1": f1(result["confusion"]),
               **{key: size[key] for key in ("raw_parameter_bytes", "declared_local_scratch_bytes",
                   "flash_delta_bytes", "static_ram_delta_bytes")},
               "inference_heap_alloc_bytes": result["heap_before_bytes"] - result["heap_after_bytes"],
               "baseline_latency_us": base["latency_mean_us"],
               "baseline_latency_p50_us": base["latency_p50_us"]}
        for metric in ("mean", "p50", "p95", "p99"):
            row[f"latency_raw_{metric}_us"] = result[f"latency_{metric}_us"]
            row[f"latency_adjusted_{metric}_us"] = (result[f"latency_{metric}_us"]
                                                       - base[f"latency_{metric}_us"])
        rows.append(row)
    # Publish only after every run, hash, and build-size pairing has passed.
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "raw_serial_logs"
    raw_dir.mkdir(exist_ok=True)
    for old in raw_dir.glob("*.log"):
        if not old.name.startswith("._") and old.exists():
            old.unlink()
    for name, content in captures:
        public = [line for line in content.splitlines() if any(marker + " " in line
                  for marker in ("TEB_BEGIN", "TEB_RESULT", "TEB_END"))]
        (raw_dir / name).write_text("\n".join(public) + "\n")
    with (output / "hardware_benchmark.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    metadata = [{**meta, "run": run, "raw_log": name} for meta, _, run, name in records]
    (output / "hardware_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (output / "baseline_timing.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ("run", "latency_mean_us", "latency_p50_us",
                                              "latency_p95_us", "latency_p99_us"), lineterminator="\n")
        writer.writeheader()
        for run, base in sorted(baselines.items()):
            writer.writerow({"run": run, **{f"latency_{key}_us": base[f"latency_{key}_us"]
                                         for key in ("mean", "p50", "p95", "p99")}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--flash", type=Path, default=OUTPUT / "flash_size.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = write_results(args.logs, args.flash, args.output)
    print(f"wrote {len(rows)} paired runs")


if __name__ == "__main__":
    main()
