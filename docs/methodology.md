# Benchmark methodology

## Source and correctness

The source dataset is `dataset/data.csv` (synthetic, seed 42). The 256-row hardware subset takes one window from each of 64 distinct test sequences per class using seed 2026 in the fixed `test` split, interleaved by class. `benchmark/export_dataset.py` emits the float32 features, ground-truth labels, and predictions from all four Python models. `tests/test_hardware_harness.py` checks that compiled host C reproduces every one of these exported predictions. The firmware repeats that comparison on the ESP32-S3 before timing and stops at the first mismatch. This is prediction parity; the hardware accuracy is against the original synthetic labels, never model-generated labels.

The board cycles through the 256 rows for 100 warm-ups and 10,000 measurements. Every timed interval consists only of `esp_timer_get_time(); predict(features); esp_timer_get_time();`. Durations are retained in static RAM, sorted afterward, and summarized with nearest-rank p50/p95/p99. A checksum is accumulated outside the timed interval and printed afterward. Interrupts are left enabled. One microsecond timer resolution and two timer calls limit precision for fast models. The figures are observed call durations, without an overhead subtraction.

## Resources

`benchmark/measure_flash.py` rebuilds the same firmware as baseline and with each one model. The baseline retains the dataset, correctness code, timer harness, result formatting, and static duration buffer but uses a trivial predictor. `flash_total_bytes` sums flashed ELF sections (`.flash.text`, `.flash.rodata`, `.flash.appdesc`, IRAM, DRAM data, and RTC sections); Flash Δ subtracts the baseline. Static RAM is `.dram0.data + .dram0.bss`, also reported as a baseline difference. The metrics come from the ELF, not a running board.

8-bit free heap before/after inference and minimum free heap since boot are queried with heap capabilities APIs. No model prediction path allocates dynamically. The minimum since boot is not an inference-only heap peak. `uxTaskGetStackHighWaterMark()` reports **bytes** in ESP-IDF 5.4.4; stack peak is configured main task stack minus its minimum remaining bytes. It includes other main-task activity before the query. These quantities cannot be combined into one "model size."

The firmware prints the CPU clock and build flags. Wi-Fi and Bluetooth are not initialized; power management is disabled in the measured configuration. These are isolated compute runs. A complete application with sensors or network activity could have different latency.

## Result provenance

Each raw serial log contains `TEB_BEGIN`, `TEB_RESULT`, and `TEB_END` JSON records. The begin record identifies chip, ESP-IDF, CPU clock, optimization, upstream Git SHA, benchmark source hash (firmware, training, export, flash, capture, and parser code), selected C model hash, sdkconfig hash, data CSV hash, and sample/iteration counts. The parser checks completeness, model identity, sample count, dataset hash, confusion matrix, and latency order before writing CSV. The raw logs remain under `results/hardware/raw_serial_logs/`. The host timestamp is in each log filename. When the Git transport is available, the firmware embeds the checked-out local HEAD; an archive-based source includes its original upstream commit plus the actual source hash.

Hardware accuracy/F1 use the 256-row subset and are distinct from the 3,000-row Python results. Power and real-sensor accuracy are unmeasured until an instrument and labeled experiment exist.
