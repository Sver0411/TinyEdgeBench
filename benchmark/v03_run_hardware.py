"""Run the clean-commit v0.3 six-variant plus timing-baseline experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.provenance import require_clean  # noqa: E402
from benchmark.run_hardware import capture, run_idf  # noqa: E402
from benchmark.v03_parse_hardware import ALL, OUTPUT, parse_log, write_results  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="exploratory only; saved under results/experimental by default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runs != 3:
        parser.error("formal protocol requires exactly three runs per variant")
    commit, dirty = require_clean(args.allow_dirty)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (ROOT / "results/experimental" / stamp if dirty else OUTPUT)
    if dirty and (output.resolve() == OUTPUT.resolve()
                  or OUTPUT.resolve() in output.resolve().parents):
        parser.error("dirty runs cannot overwrite formal v0.3 results")
    with tempfile.TemporaryDirectory(prefix="tinyedge-v03-") as temp:
        temp_dir = Path(temp)
        flash = temp_dir / "flash_size.csv"
        subprocess.run([sys.executable, str(ROOT / "benchmark/v03_measure_flash.py"),
                        "--output", str(flash)], check=True)
        paths = []
        for variant in ALL:
            print(f"building and flashing {variant}", flush=True)
            run_idf(f"-DTINYEDGEBENCH_MODEL={variant}", "build")
            run_idf("-p", args.port, "flash")
            for run in range(1, args.runs + 1):
                log = capture(args.port)
                path = temp_dir / f"{variant}_{run}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
                path.write_text(log)
                meta, _ = parse_log(log)
                if meta["model"] != variant or meta["git_commit"] != commit or meta["git_dirty"] != dirty:
                    raise RuntimeError(f"unexpected firmware provenance for {variant} run {run}")
                paths.append(path)
                print(f"captured {variant} run {run}", flush=True)
        write_results(paths, flash, output, allow_dirty=dirty)
        shutil.copyfile(flash, output / "flash_size.csv")
    print(f"v0.3 evidence saved to {output}", flush=True)


if __name__ == "__main__":
    main()
