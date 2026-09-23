# ESP32-S3 hardware protocol

1. Verify the port with `python -m esptool --port PORT chip_id` and `python -m esptool --port PORT flash_id`. Confirm ESP32-S3 and record the detected Flash size. The tested board exposes `/dev/cu.usbmodem5C930836161`; port names vary.
2. Activate ESP-IDF v5.4+, then run `python -m pytest tests/ -q`, `python training/export_models.py`, and `python benchmark/export_dataset.py`.
3. Run `python benchmark/measure_flash.py`. It builds baseline plus four single-model images and records both Flash and static RAM from the ELF.
4. Run `python benchmark/run_hardware.py --port PORT --runs 3`. The script builds, flashes, resets, captures three serial runs per model, validates each run, and writes CSV, JSON metadata, and raw logs. Any `TEB_ERROR` or incomplete capture stops the script; inspect the saved raw log and fix correctness before repeating.
5. Run `python benchmark/run_benchmark.py`, `python benchmark/plot_hardware.py`, and `python benchmark/update_readme.py`. The README tables are generated from the CSV files.

The firmware uses the same 256 samples, 100 warm-ups, 10,000 measurements, and one inference path for every model. Default target is ESP32-S3; `firmware/sdkconfig.defaults` sets a 16 MB image header, 160 MHz CPU, and performance optimization. Check actual board details rather than applying this configuration to an unknown chip.

Each run writes `TEB_BEGIN`, `TEB_RESULT`, `TEB_END` lines. To parse an already saved serial capture, use `python benchmark/parse_hardware_log.py path/to/serial.log`. Keep complete, unedited logs with the resulting CSV. Measurements from different configurations must not be pooled.
