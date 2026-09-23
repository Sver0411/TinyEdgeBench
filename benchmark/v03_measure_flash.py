"""Measure single-predictor v0.3 builds without touching v0.2 results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.measure_flash import (BUILD_DIR, FIRMWARE, FLASH_SECTIONS, ROOT,
                                     MISSING_IDF_MESSAGE, build_and_size,
                                     find_project_name, find_size_tool, how_to_run_idf)

VARIANTS = ("rule", "tree", "logistic_fp32", "logistic_int8", "mlp_fp32", "mlp_int8")
OUTPUT = ROOT / "results/v0.3/flash_size.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--idf-path", default=None)
    args = parser.parse_args()
    prefix = how_to_run_idf(args.idf_path)
    if prefix is None and not shutil.which("idf.py"):
        raise SystemExit(MISSING_IDF_MESSAGE)
    tool = find_size_tool("esp32s3", None)
    if tool is None:
        raise SystemExit("xtensa-esp32s3-elf-size unavailable")
    project = find_project_name()
    measured = {}
    for variant in ("baseline", *VARIANTS):
        print(f"building v0.3 variant: {variant}", flush=True)
        measured[variant] = build_and_size(variant, project, idf_prefix=prefix, size_tool=tool)
    for key in ("git_commit", "git_dirty", "source_sha256", "sdkconfig_sha256",
                "quantization_sha256"):
        if len({str(p[key]) for _, p in measured.values()}) != 1:
            raise SystemExit(f"mixed build provenance: {key}")
    def flash(sections):
        return sum(sections.get(name, 0) for name in FLASH_SECTIONS)
    baseline = measured["baseline"][0]
    base_ram = baseline.get("dram0.data", 0) + baseline.get("dram0.bss", 0)
    with (ROOT / "results/v0.3/offline_quantization.csv").open(newline="") as handle:
        offline = {row["variant"]: row for row in csv.DictReader(handle)}
    rows = []
    for variant in VARIANTS:
        sections, provenance = measured[variant]
        ram = sections.get("dram0.data", 0) + sections.get("dram0.bss", 0)
        rows.append({
            "variant": variant,
            "model_family": offline[variant]["model_family"],
            "precision": offline[variant]["precision"],
            "implementation": "portable_scalar",
            **{key: provenance[key] for key in ("git_commit", "git_dirty", "source_sha256",
                     "sdkconfig_sha256", "model_sha256", "quantization_sha256")},
            "flash_total_bytes": flash(sections),
            "flash_delta_bytes": flash(sections) - flash(baseline),
            "static_ram_bytes": ram,
            "static_ram_delta_bytes": ram - base_ram,
            "raw_parameter_bytes": offline[variant]["raw_parameter_bytes"],
            "declared_local_scratch_bytes": offline[variant]["declared_local_scratch_bytes"],
        })
    print("restoring default build", flush=True)
    build_and_size("rule", project, idf_prefix=prefix, size_tool=tool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
