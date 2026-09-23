"""Build, flash and capture repeated ESP32-S3 runs for all four models.

Run in an activated ESP-IDF shell:
    python benchmark/run_hardware.py --port /dev/cu.usbmodem... --runs 3
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"
sys.path.insert(0, str(ROOT))
from benchmark.parse_hardware_log import MODELS, parse_log, write_results  # noqa: E402
from benchmark.provenance import require_clean  # noqa: E402


def run_idf(*args):
    subprocess.run(["idf.py", *args], cwd=FIRMWARE, check=True)


def capture(port, timeout=30):
    import serial

    lines = []
    with serial.Serial(port, 115200, timeout=0.5) as device:
        # Opening first, then resetting ensures the complete benchmark is read.
        device.dtr = False
        device.rts = True
        time.sleep(0.12)
        device.rts = False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = device.readline().decode("utf-8", errors="replace")
            if line:
                lines.append(line)
                if "TEB_END " in line or "TEB_ERROR " in line:
                    break
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="exploratory run only; mark results git_dirty=true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    commit, dirty = require_clean(args.allow_dirty)
    with tempfile.TemporaryDirectory(prefix="tinyedge-hardware-") as temp:
        temp_dir = Path(temp)
        sizes = temp_dir / "flash_size.csv"
        subprocess.run([sys.executable, str(ROOT / "benchmark/measure_flash.py"),
                        "--output", str(sizes)], check=True)
        paths = []
        for model in args.models:
            print(f"building and flashing {model}", flush=True)
            run_idf(f"-DTINYEDGEBENCH_MODEL={model}", "build")
            run_idf("-p", args.port, "flash")
            for run in range(1, args.runs + 1):
                log = capture(args.port)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                path = temp_dir / f"{model}_{run}_{stamp}.log"
                path.write_text(log)
                records = parse_log(log)
                if len(records) != 1 or records[0]["metadata"]["model"] != model:
                    raise RuntimeError(f"unexpected serial output; saved {path}")
                meta = records[0]["metadata"]
                if meta["git_commit"] != commit or meta["git_dirty"] != dirty:
                    raise RuntimeError("hardware firmware Git state differs from the run")
                paths.append(path)
                print(f"captured {model} run {run}: {path.name}", flush=True)
        write_results(paths, build_sizes=sizes)
        shutil.copyfile(sizes, ROOT / "results/flash_size.csv")
    print("hardware results saved to results/hardware/", flush=True)


if __name__ == "__main__":
    main()
