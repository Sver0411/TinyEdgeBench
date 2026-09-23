# Benchmark methodology

## Source and correctness

The source dataset is `dataset/data.csv` (synthetic, seed 42). The 256-row hardware subset takes one window from each of 64 distinct test sequences per class using seed 2026 in the fixed `test` split, interleaved by class. `benchmark/export_dataset.py` emits the float32 features, ground-truth labels, and predictions from all four Python models. `tests/test_hardware_harness.py` checks that compiled host C reproduces every one of these exported predictions. The firmware repeats that comparison on the ESP32-S3 before timing and stops at the first mismatch. This is prediction parity; the hardware accuracy is against the original synthetic labels, never model-generated labels.

The board cycles through the 256 rows for 100 warm-ups and 2,000 timed batches of 64 predictions (128,000 calls). Each timed interval brackets the 64 calls with `esp_timer_get_time()`. Batch durations are retained in static RAM, sorted afterward, and summarized with nearest-rank p50/p95/p99, then divided by 64 for fractional microseconds per inference. A checksum keeps predictions observable and is printed afterward. Interrupts are left enabled. The figures include timer overhead; batch percentiles describe average calls within batches, not individual call tails.

## Resources

`benchmark/measure_flash.py` rebuilds the same firmware as baseline and with each one model. The baseline retains the dataset, correctness code, timer harness, result formatting, and static duration buffer but uses a trivial predictor. `flash_total_bytes` sums flashed ELF sections (`.flash.text`, `.flash.rodata`, `.flash.appdesc`, IRAM, DRAM data, and RTC sections); Flash Δ subtracts the baseline. Static RAM is `.dram0.data + .dram0.bss`, also reported as a baseline difference. The metrics come from the ELF, not a running board.

8-bit free heap before/after inference and minimum free heap since boot are queried with heap capabilities APIs. The before/after difference is the top-table inference heap allocation. No model prediction path allocates dynamically. The minimum since boot is not an inference-only heap peak. `uxTaskGetStackHighWaterMark()` reports **bytes** in ESP-IDF 5.4.4; the main task value is retained as a diagnostic in the CSV. It includes other main-task activity and is omitted from the model comparison table.

The firmware prints the CPU clock and build flags. Wi-Fi and Bluetooth are not initialized; power management is disabled in the measured configuration. These are isolated compute runs. A complete application with sensors or network activity could have different latency.

## Result provenance

Each public serial log contains the `TEB_BEGIN`, `TEB_RESULT`, and `TEB_END` JSON records. The begin record identifies chip, ESP-IDF, CPU clock, optimization, local Git HEAD, dirty state, benchmark source hash (firmware, training, export, flash, capture, provenance, and parser code), selected C model hash, sdkconfig hash, data CSV hash, and batch counts. The Flash CSV includes the same build provenance. The parser requires exact equality of Git commit, dirty state, source hash, model hash, and sdkconfig hash before merging sizes. It also checks completeness, model identity, dataset hash, confusion matrix, and latency order. Public logs retain only benchmark records, excluding boot lines that may carry a MAC. The host timestamp is in each log filename. Formal measurement requires a clean commit; dirty runs are explicitly marked and cannot feed the README. The code commit precedes measurement, so the final results publication commit is expected to differ from the embedded measurement commit.

The top table uses full 3,000-row Python accuracy/F1; hardware CSV accuracy/F1 describes the 256-row parity subset. Power and real-sensor accuracy are unmeasured until an instrument and labeled experiment exist.
