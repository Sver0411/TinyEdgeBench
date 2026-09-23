"""Generate v0.3 comparison tables from committed offline and hardware CSVs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v0.3"
ORDER = ("rule", "tree", "logistic_fp32", "logistic_int8", "mlp_fp32", "mlp_int8")
LABEL = {"rule": "Rule", "tree": "Tree", "logistic_fp32": "Logistic",
         "logistic_int8": "Logistic", "mlp_fp32": "MLP", "mlp_int8": "MLP"}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def format_latency(rows, key):
    values = [float(row[key]) for row in rows]
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def tables():
    offline = {row["variant"]: row for row in read_csv(OUT / "offline_quantization.csv")}
    hardware = read_csv(OUT / "hardware_benchmark.csv")
    sizes = {row["variant"]: row for row in read_csv(OUT / "flash_size.csv")}
    metadata = json.loads((OUT / "hardware_metadata.json").read_text())
    if (len(metadata) != 21 or len(hardware) != 18
            or any(item["git_dirty"] for item in metadata)
            or any(row["git_dirty"].lower() != "false" for row in hardware)):
        raise ValueError("v0.3 README requires complete clean hardware results")
    if len({item["git_commit"] for item in metadata}) != 1:
        raise ValueError("mixed v0.3 measurement commits")
    grouped = {variant: [row for row in hardware if row["variant"] == variant]
               for variant in ORDER}
    if any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("need three hardware runs per variant")
    for variant in ORDER:
        if any(row["git_commit"] != sizes[variant]["git_commit"]
               or row["quantization_sha256"] != sizes[variant]["quantization_sha256"]
               for row in grouped[variant]):
            raise ValueError("build-size provenance differs")
    main = ["| Model | Precision | Test accuracy | Macro F1 | Flash Δ | Raw params | Raw p50 µs | Raw p95 µs |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in ORDER:
        o, s, rows = offline[variant], sizes[variant], grouped[variant]
        raw = ("N/A" if o["raw_parameter_bytes"] == "N/A"
               else f"{int(o['raw_parameter_bytes'])} B")
        main.append(f"| {LABEL[variant]} | {o['precision'].upper()} | "
                    f"{float(o['accuracy']):.4f} | {float(o['macro_f1']):.4f} | "
                    f"{int(s['flash_delta_bytes']):,} B | {raw} | "
                    f"{format_latency(rows, 'latency_raw_p50_us')} | "
                    f"{format_latency(rows, 'latency_raw_p95_us')} |")
    compare = ["| Pair | Δ accuracy | Δ Macro F1 | Flash saving | Raw p50 change |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for family in ("logistic", "mlp"):
        fp, q = f"{family}_fp32", f"{family}_int8"
        flash_fp, flash_q = int(sizes[fp]["flash_delta_bytes"]), int(sizes[q]["flash_delta_bytes"])
        lat_fp, lat_q = mean(grouped[fp], "latency_raw_p50_us"), mean(grouped[q], "latency_raw_p50_us")
        saving = flash_fp - flash_q
        delta = lat_q - lat_fp
        compare.append(f"| {family.title()} INT8 − FP32 | "
                       f"{float(offline[q]['accuracy']) - float(offline[fp]['accuracy']):+.4f} | "
                       f"{float(offline[q]['macro_f1']) - float(offline[fp]['macro_f1']):+.4f} | "
                       f"{saving:+,} B ({saving / flash_fp * 100:+.1f}%) | "
                       f"{delta:+.3f} µs ({delta / lat_fp * 100:+.1f}%) |")
    return "\n".join(main), "\n".join(compare)


def replace_block(content, marker, table):
    start, end = f"<!-- {marker}_START -->", f"<!-- {marker}_END -->"
    updated, count = re.subn(re.escape(start) + r".*?" + re.escape(end),
                             start + "\n" + table + "\n" + end, content, flags=re.S)
    if count != 1:
        raise ValueError(f"missing {marker} marker")
    return updated


def main():
    main_table, comparison = tables()
    for name in ("README.md", "README.zh-CN.md"):
        path = ROOT / name
        text = replace_block(path.read_text(), "V03_MAIN", main_table)
        text = replace_block(text, "V03_COMPARE", comparison)
        path.write_text(text)
        print(f"updated {name}")


if __name__ == "__main__":
    main()
