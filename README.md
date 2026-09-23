# TinyEdgeBench v0.2

**English** | [简体中文](README.zh-CN.md)

TinyEdgeBench compares four small classifiers on one sensor-window task, using the same eight features, the same test rows, and one ESP32-S3 board. It asks whether extra classification quality is worth the compiled Flash, static RAM, stack, heap, and inference time.

<!-- HARDWARE_TABLE_START -->
| Model | Accuracy | Macro F1 | Flash Δ | Static RAM Δ | Stack peak | Heap Δ | p50 µs | p95 µs | p99 µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | 0.6719 | 0.6331 | 40 B | 0 B | 1616 B | 0 B | 2.0 ± 0.0 | 2.0 ± 0.0 | 2.0 ± 0.0 |
| Logistic | 0.8516 | 0.8508 | 204 B | 0 B | 1616 B | 0 B | 3.0 ± 0.0 | 4.0 ± 0.0 | 4.0 ± 0.0 |
| Tree | 0.8164 | 0.8142 | 284 B | 0 B | 1616 B | 0 B | 1.0 ± 0.0 | 2.0 ± 0.0 | 2.0 ± 0.0 |
| MLP | 0.8438 | 0.8420 | 944 B | 0 B | 1616 B | 0 B | 9.0 ± 0.0 | 10.0 ± 0.0 | 10.0 ± 0.0 |
<!-- HARDWARE_TABLE_END -->

The accuracy and F1 in this table are **hardware predictions on the fixed 256-row synthetic test subset**. Flash and static RAM are build measurements. Latency, heap, and stack come from the physical ESP32-S3. [Raw logs and machine-readable results](results/hardware/) are the source of this table; `benchmark/update_readme.py` generates it. Power: **Not measured yet**.

## Task and models

The task classifies the last eight sensor readings as `NORMAL`, `RAPID_CHANGE`, `SLOW_DRIFT`, or `NOISY`. All four methods share `temp`, `hum`, `light`, their latest changes, `motion`, and `drift`:

| Model | Implementation |
| --- | --- |
| Rule | Five fixed thresholds on motion and drift |
| Logistic | StandardScaler + multinomial logistic regression |
| Tree | Decision tree, depth at most five, exported as branches |
| MLP | StandardScaler + 8→8→8→4 ReLU network |

`dataset/generate.py` produces 20,000 **synthetic** rows with seed 42. A sequence stays entirely within train, validation, or test. The full test split contains 3,000 rows. `benchmark/export_dataset.py` takes one window from each of 64 distinct test sequences per class with a fixed seed, interleaves them, and exports 256 float32 feature rows, ground-truth labels, and Python prediction oracles to C. This fixed subset is used by every hardware build. The training data and test subset are never changed for an individual model.

## Offline results: full test split

<!-- OFFLINE_TABLE_START -->
| Model | Accuracy | Macro F1 | Raw constants |
| --- | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 20 B |
| Logistic | 0.8283 | 0.8256 | 208 B |
| Tree | 0.8203 | 0.8167 | N/A (control flow) |
| MLP | 0.8250 | 0.8214 | 784 B |
<!-- OFFLINE_TABLE_END -->

These accuracy and macro F1 values come from Python on 3,000 synthetic test rows. Raw constants count numeric float32 thresholds, weights, biases, and scaler values; a tree exported as control flow has no comparable parameter array. [The full offline CSV](results/benchmark.csv) and [confusion matrix CSV](results/confusion.csv) retain the unrounded counts.

## Measurements and scope

- **Build:** `benchmark/measure_flash.py` builds baseline, rule, logistic, tree, and MLP separately with the same dataset and measurement harness. Flash Δ is the sum of flashed ELF sections minus the baseline. Static RAM Δ is `.data + .bss` minus the same baseline. Both include small predictor selection differences and are not pure parameter sizes. [Build CSV](results/flash_size.csv).
- **Hardware correctness:** Before timing, the board checks all 256 C predictions against the Python oracle. A mismatch aborts the run. The board also reports labels, correct count, confusion matrix, accuracy input, and checksum.
- **Hardware latency:** `esp_timer_get_time()` brackets only the C predictor call. There are 100 warm-up calls and 10,000 timed calls, cycling over all 256 samples. No serial output, allocation, or logging occurs in the timed interval. The volatile checksum keeps predictions observable. The reported min, mean, p50, p95, p99, and max include timer-call overhead and 1 µs clock quantization; no overhead is subtracted.
- **Runtime RAM:** 8-bit free heap before and after inference and the minimum free heap since boot are reported separately. The predictor has no dynamic allocation. The main task stack high-water mark is in **bytes** in ESP-IDF 5.4.4; estimated peak is configured stack minus minimum remaining. It covers the task through the measurement, not a model-only stack frame.
- **Environment:** One model is linked per build. The firmware does not initialize Wi-Fi or Bluetooth. CPU frequency, ESP-IDF version, optimization, logging level, source hash, model hash, configuration hash, upstream Git commit, dataset hash, and run settings are printed by the board. The host parser rejects incomplete logs and dataset mismatches. This is an isolated compute benchmark, not full application latency.

See [methodology](docs/methodology.md) and [hardware protocol](docs/hardware_protocol.md) for exact definitions and commands.

## Reproduce

Use an ESP-IDF v5.4+ shell and Python dependencies from `requirements.txt`:

```bash
python -m pytest tests/ -q
python training/export_models.py
python benchmark/export_dataset.py
python benchmark/measure_flash.py
python benchmark/run_benchmark.py
python benchmark/run_hardware.py --port /dev/cu.usbmodemYOURBOARD --runs 3
python benchmark/plot_hardware.py
python benchmark/update_readme.py
```

`run_hardware.py` builds, flashes, resets, captures serial, validates the records, and writes `results/hardware/hardware_benchmark.csv`, `hardware_metadata.json`, and `raw_serial_logs/`. For an existing capture use `python benchmark/parse_hardware_log.py serial.log`. `idf.py -DTINYEDGEBENCH_MODEL=rule build` (or `logistic`, `tree`, `mlp`) builds one model manually. `baseline` is only a resource sizing reference.

## Limits

- The labels and accuracy are from synthetic data. Real sensor traces and a labeled real-world accuracy experiment: **Not measured yet**.
- Power and energy: **Not measured yet**; no current-measurement instrument was identified.
- FP32 only, one dataset and seed, one board configuration. INT8 is deferred until this FP32 baseline is stable.
- The 256-row hardware subset is not the full 3,000-row test split, so its accuracy differs from offline accuracy.
- At microsecond scale, timer resolution and overhead limit latency comparisons between close models. Repeated runs and p95/p99 expose some scheduling variation; interrupts are not disabled.

The existing [offline accuracy/Flash plot](results/accuracy_vs_model_size.png) and [confusion matrices](results/confusion_matrices.png) remain available. Hardware plots are generated only from validated runs in `results/hardware/`.

## License

MIT. See [LICENSE](LICENSE).
