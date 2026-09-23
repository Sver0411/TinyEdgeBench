"""Regenerate the English and Chinese result tables from committed CSV files."""

import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("rule", "logistic", "tree", "mlp")
LABELS = {"rule": "Rule", "logistic": "Logistic", "tree": "Tree", "mlp": "MLP"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def format_runs(rows, key, digits=1):
    values = [float(row[key]) for row in rows]
    return f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"


def build_tables():
    offline = {row["method"]: row for row in read_csv(ROOT / "results/benchmark.csv")}
    hardware = read_csv(ROOT / "results/hardware/hardware_benchmark.csv")
    metadata = json.loads((ROOT / "results/hardware/hardware_metadata.json").read_text())
    if (not metadata or any(item["git_dirty"] for item in metadata)
            or any(row.get("git_dirty", "").lower() != "false" for row in hardware)):
        raise ValueError("README requires clean committed hardware measurements")
    grouped = {model: [row for row in hardware if row["model"] == model] for model in MODELS}
    if any(len(rows) < 3 for rows in grouped.values()):
        raise ValueError("need at least three hardware runs per model")
    if any(any(not row["flash_delta_bytes"] or not row["static_ram_delta_bytes"]
                   for row in rows) for rows in grouped.values()):
        raise ValueError("run benchmark/measure_flash.py and reparse logs first")
    hardware_lines = [
        "| Model | Accuracy (3,000) | Macro F1 (3,000) | Flash Δ | Static RAM Δ | Inference heap alloc | p50 µs | p95 µs | p99 µs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    offline_lines = ["| Model | Accuracy | Macro F1 | Raw constants |",
                     "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        rows = grouped[model]
        reference = rows[0]
        if any(row["accuracy"] != reference["accuracy"] or row["macro_f1"] != reference["macro_f1"]
               for row in rows):
            raise ValueError(f"quality metrics differ across hardware runs for {model}")
        o = offline[model]
        heap_delta = statistics.mean(int(row["heap_before_bytes"])
                                     - int(row["heap_after_bytes"]) for row in rows)
        hardware_lines.append(
            f"| {LABELS[model]} | {float(o['accuracy']):.4f} | "
            f"{float(o['macro_f1']):.4f} | {int(reference['flash_delta_bytes']):,} B | "
            f"{int(reference['static_ram_delta_bytes']):,} B | {heap_delta:.0f} B | {format_runs(rows, 'latency_p50_us')} | "
            f"{format_runs(rows, 'latency_p95_us')} | {format_runs(rows, 'latency_p99_us')} |"
        )
        raw = f"{o['raw_constants_bytes']} B" if o["raw_constants_bytes"] else "N/A (control flow)"
        offline_lines.append(f"| {LABELS[model]} | {float(o['accuracy']):.4f} | "
                             f"{float(o['macro_f1']):.4f} | {raw} |")
    return "\n".join(hardware_lines), "\n".join(offline_lines)


def replace_block(content, name, table):
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    pattern = re.escape(start) + r".*?" + re.escape(end)
    replacement = start + "\n" + table + "\n" + end
    updated, count = re.subn(pattern, replacement, content, flags=re.S)
    if count != 1:
        raise ValueError(f"missing or duplicate {name} marker")
    return updated


def main():
    hardware, offline = build_tables()
    for name in ("README.md", "README.zh-CN.md"):
        path = ROOT / name
        content = replace_block(path.read_text(), "HARDWARE_TABLE", hardware)
        content = replace_block(content, "OFFLINE_TABLE", offline)
        path.write_text(content)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
