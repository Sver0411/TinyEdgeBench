"""Measure the compiled flash footprint of each predictor.

Builds the firmware five times with the CMake option
`-DTINYEDGEBENCH_MODEL=<baseline|rule|logistic|tree|mlp>` and reads the ELF
section sizes straight from the ESP-IDF toolchain:

    results/flash_size.csv

Every build differs only in which single predictor source file is linked, so

    flash_delta = flash_bytes(model build) - flash_bytes(baseline build)

is the real cost of that predictor's compiled C code - nothing is estimated.

Run:
    python benchmark/measure_flash.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"
RESULTS = ROOT / "results"
BUILD_DIR = FIRMWARE / "build"

VARIANTS = ("baseline", "rule", "logistic", "tree", "mlp")
RESTORE_VARIANT = "all"
TARGET = "esp32s3"

# Sections that make up the flashed image (their contents all live in flash;
# some are copied to RAM at boot). Dummy and debug sections are excluded.
FLASH_SECTIONS = (
    "flash.text", "flash.rodata", "flash.appdesc",
    "iram0.text", "iram0.vectors",
    "dram0.data",
    "rtc.text", "rtc.force_fast", "rtc.force_slow", "rtc_reserved",
)

DEFAULT_EXPORT = "/Users/mac/esp/esp-idf/export.sh"
DEFAULT_TOOLS = "/Users/mac/.espressif"
DEFAULT_PYTHON_BIN = "/Users/mac/.workbuddy/binaries/python/versions/3.13.12/bin"


def find_project_name():
    match = re.search(r"project\(\s*([A-Za-z0-9_\-]+)\s*\)", (FIRMWARE / "CMakeLists.txt").read_text())
    return match.group(1) if match else "app"


def build_and_size(variant, project, *, export, tools, python_bin):
    """Build one variant and return {section name: size} from the toolchain."""
    elf = BUILD_DIR / f"{project}.elf"
    script = f"""
set -e
source {export} >/dev/null 2>&1
cd {FIRMWARE}
idf.py -DTINYEDGEBENCH_MODEL={variant} build
SIZE=$(command -v xtensa-{TARGET}-elf-size || true)
if [ -z "$SIZE" ]; then
    SIZE=$(ls {tools}/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/xtensa-{TARGET}-elf-size | head -1)
fi
"$SIZE" -A {elf}
"""
    env = {
        "HOME": os.environ.get("HOME", "/root"),
        "PATH": f"{python_bin}:/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin",
        "IDF_TOOLS_PATH": tools,
        "TERM": os.environ.get("TERM", "xterm"),
    }
    result = subprocess.run(["/bin/bash", "-c", script], capture_output=True, text=True, env=env)
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
    parser.add_argument("--export", default=DEFAULT_EXPORT, help="path to ESP-IDF export.sh")
    parser.add_argument("--tools", default=DEFAULT_TOOLS, help="IDF_TOOLS_PATH")
    parser.add_argument("--python-bin", default=DEFAULT_PYTHON_BIN,
                        help="directory of the python3 the IDF env was installed with")
    args = parser.parse_args()

    if not Path(args.export).exists():
        raise SystemExit(f"ESP-IDF not found at {args.export} - nothing to measure")

    RESULTS.mkdir(parents=True, exist_ok=True)
    project = find_project_name()

    measured = {}
    for variant in VARIANTS:
        print(f"building variant: {variant}", flush=True)
        measured[variant] = build_and_size(variant, project, export=args.export,
                                           tools=args.tools, python_bin=args.python_bin)

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
    build_and_size(RESTORE_VARIANT, project, export=args.export,
                   tools=args.tools, python_bin=args.python_bin)


if __name__ == "__main__":
    main()
