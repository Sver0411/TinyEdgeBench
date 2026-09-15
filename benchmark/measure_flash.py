"""Measure the compiled flash footprint of each predictor.

Builds the firmware five times with the CMake option
`-DTINYEDGEBENCH_MODEL=<baseline|rule|logistic|tree|mlp>` and reads the ELF
section sizes from the ESP-IDF toolchain:

    results/flash_size.csv

Each build differs only in which single predictor source file is linked, so

    flash_delta = flash_bytes(model build) - flash_bytes(baseline build)

is a compiled flash-section delta for that predictor. It covers the predictor
plus the small amount of demo / scaffolding code that any non-empty build pulls
in, so it is suitable for comparing the four C implementations against each
other, but it is not a pure "size of the model" figure.

ESP-IDF discovery order (nothing machine-specific is assumed):

    1. IDF_PATH set          -> source $IDF_PATH/export.sh inside the build
    2. idf.py already on PATH (activated shell) -> use it directly
    3. neither               -> error out

Run:
    python benchmark/measure_flash.py                 # after export.sh, or with IDF_PATH set
    python benchmark/measure_flash.py --idf-path ...  # explicit override
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"
RESULTS = ROOT / "results"
BUILD_DIR = FIRMWARE / "build"

VARIANTS = ("baseline", "rule", "logistic", "tree", "mlp")
RESTORE_VARIANT = "all"

MISSING_IDF_MESSAGE = (
    "ESP-IDF environment not found.\n"
    "Run ESP-IDF export.sh first or set IDF_PATH."
)

# Sections that make up the flashed image (their contents all live in flash;
# some are copied to RAM at boot). Dummy and debug sections are excluded.
FLASH_SECTIONS = (
    "flash.text", "flash.rodata", "flash.appdesc",
    "iram0.text", "iram0.vectors",
    "dram0.data",
    "rtc.text", "rtc.force_fast", "rtc.force_slow", "rtc_reserved",
)


def find_project_name():
    match = re.search(r"project\(\s*([A-Za-z0-9_\-]+)\s*\)", (FIRMWARE / "CMakeLists.txt").read_text())
    return match.group(1) if match else "app"


def how_to_run_idf(idf_path):
    """Return the shell prefix that makes idf.py available, or None."""
    if idf_path:
        export = Path(idf_path) / "export.sh"
        if export.is_file():
            return f"source {export} >/dev/null 2>&1"
        raise SystemExit(f"IDF_PATH={idf_path} does not contain export.sh")
    if shutil.which("idf.py"):
        return None
    return None


def find_size_tool(target, tools_path):
    """Locate the toolchain size binary, preferring an already activated PATH."""
    name = f"xtensa-{target}-elf-size"
    found = shutil.which(name)
    if found:
        return found
    search_root = tools_path or os.environ.get("IDF_TOOLS_PATH")
    if search_root:
        matches = sorted(glob.glob(
            f"{search_root}/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/{name}"))
        if matches:
            return matches[-1]
    return None


def build_and_size(variant, project, *, idf_prefix, size_tool):
    """Build one variant and return {section name: size} from the toolchain."""
    elf = BUILD_DIR / f"{project}.elf"
    lines = ["set -e"]
    if idf_prefix:
        lines.append(idf_prefix)
    lines += [
        f"cd {FIRMWARE}",
        f"idf.py -DTINYEDGEBENCH_MODEL={variant} build",
        f'"{size_tool}" -A {elf}',
    ]
    result = subprocess.run(["/bin/bash", "-c", "\n".join(lines)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr[-4000:])
        raise SystemExit(f"build failed for variant '{variant}'")

    sections = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].startswith("."):
            sections[parts[0].lstrip(".")] = int(parts[1])
    if "flash.text" not in sections:
        raise SystemExit(f"could not read ELF sections from {elf}")
    return sections


def main():
    parser = argparse.ArgumentParser(description="Measure compiled flash footprint per predictor.")
    parser.add_argument("--idf-path", default=os.environ.get("IDF_PATH"),
                        help="ESP-IDF checkout (defaults to $IDF_PATH)")
    parser.add_argument("--target", default="esp32s3", help="IDF target, e.g. esp32s3")
    parser.add_argument("--tools-path", default=os.environ.get("IDF_TOOLS_PATH"),
                        help="IDF_TOOLS_PATH (defaults to $IDF_TOOLS_PATH)")
    args = parser.parse_args()

    idf_prefix = how_to_run_idf(args.idf_path)
    if idf_prefix is None and not shutil.which("idf.py"):
        raise SystemExit(MISSING_IDF_MESSAGE)

    size_tool = find_size_tool(args.target, args.tools_path)
    if size_tool is None:
        raise SystemExit(
            f"could not find xtensa-{args.target}-elf-size.\n"
            "Activate an ESP-IDF environment or set IDF_TOOLS_PATH."
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    project = find_project_name()

    measured = {}
    for variant in VARIANTS:
        print(f"building variant: {variant}", flush=True)
        measured[variant] = build_and_size(variant, project,
                                           idf_prefix=idf_prefix, size_tool=size_tool)

    def total(sections):
        return sum(sections.get(name, 0) for name in FLASH_SECTIONS)

    baseline = total(measured["baseline"])
    rows = []
    for variant in VARIANTS[1:]:
        sections = measured[variant]
        rows.append({
            "method": variant,
            "text_bytes": sections.get("flash.text", 0),
            "rodata_bytes": sections.get("flash.rodata", 0),
            "data_bytes": sections.get("dram0.data", 0),
            "iram_text_bytes": sections.get("iram0.text", 0),
            "flash_total_bytes": total(sections),
            "flash_delta_bytes": total(sections) - baseline,
        })

    out_path = RESULTS / "flash_size.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nbaseline (no predictor): {baseline} bytes of flash image")
    print(f"{'method':10s} {'text':>8s} {'rodata':>8s} {'data':>8s} {'total':>8s} {'delta':>8s}")
    for row in rows:
        print(f"{row['method']:10s} {row['text_bytes']:8d} {row['rodata_bytes']:8d} "
              f"{row['data_bytes']:8d} {row['flash_total_bytes']:8d} {row['flash_delta_bytes']:8d}")
    print(f"\nwrote {out_path}")

    # leave the build tree back in its default configuration (all predictors)
    print("restoring default build (all four predictors)", flush=True)
    build_and_size(RESTORE_VARIANT, project, idf_prefix=idf_prefix, size_tool=size_tool)


if __name__ == "__main__":
    main()
