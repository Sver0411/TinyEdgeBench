# ESP32-S3 hardware protocol

1. Verify the port with `python -m esptool --port PORT chip_id` and `python -m esptool --port PORT flash_id`. Confirm ESP32-S3 and record the detected Flash size. Port names vary; do not commit unique serial identifiers or MAC addresses.
2. Activate ESP-IDF v5.4+, then run `python -m pytest tests/ -q`, `python training/export_models.py`, and `python benchmark/export_dataset.py`.
3. Commit all code and generated models; confirm a clean Git working tree. `run_hardware.py` rejects dirty state by default. Use `--allow-dirty` only for exploratory runs; their metadata records `git_dirty=true` and cannot populate the README.
4. Run `python benchmark/run_hardware.py --port PORT --runs 3`. The script measures baseline and four single-model Flash images, then builds, flashes, resets, and captures three serial runs per model. It validates each run and exact build-size provenance before writing CSV, JSON metadata, and sanitized public logs. A failed capture stops the script before publishing results.
5. Run `python benchmark/run_benchmark.py`, `python benchmark/plot_hardware.py`, and `python benchmark/update_readme.py`. The README tables are generated from the CSV files.

The firmware uses the same 256 samples, 100 warm-ups, 2,000 batches of 64 predictions, and one inference path for every model. Default target is ESP32-S3; `firmware/sdkconfig.defaults` sets a 16 MB image header, 160 MHz CPU, and performance optimization. Check actual board details rather than applying this configuration to an unknown chip.

Each run writes `TEB_BEGIN`, `TEB_RESULT`, `TEB_END` lines. To parse an existing capture, use `python benchmark/parse_hardware_log.py path/to/serial.log --build-sizes path/to/flash_size.csv`. Keep the complete original capture privately if needed; committed logs contain only benchmark records. Measurements from different commits or configurations must not be pooled. Publish results in a separate commit after the clean code commit used for measurement.
