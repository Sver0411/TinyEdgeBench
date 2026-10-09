<h1 align="center">TinyEdgeBench</h1>

<p align="center"><strong>Lightweight Classifiers and FP32 / INT8 Tradeoffs on ESP32-S3</strong></p>

<p align="center">Shared Data · Quantization · Measured Resources</p>

---

<p align="center">
  <a href="#project-overview">Overview</a> ·
  <a href="#research-question-and-method">Method</a> ·
  <a href="#implementation-and-measurement-status">Status</a> ·
  <a href="#relationship-to-smart-agriculture-edge-ai">Related System</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

<a id="project-overview"></a>

TinyEdgeBench compares Rule, Logistic, Tree, and MLP classifiers on one sensor-window task with shared data and physical ESP32-S3 measurements. v0.3 examines the Flash, latency, and accuracy tradeoffs of post-training INT8 quantization for Logistic and MLP; Rule and Tree remain native controls.

## Research question and method

Does a smaller numerical representation improve practical MCU inference? The task classifies windows of eight sensor readings as `NORMAL`, `RAPID_CHANGE`, `SLOW_DRIFT`, or `NOISY`, using eight shared features: temperature, humidity, light, their latest changes, `motion`, and `drift`.

| Model | Implementation |
| --- | --- |
| Rule | Five fixed thresholds on motion and drift |
| Logistic | StandardScaler + multinomial logistic regression |
| Tree | Decision tree, maximum depth five, exported as branches |
| MLP | StandardScaler + 8→8→8→4 ReLU network |

The dataset contains 20,000 synthetic rows (seed 42), with sequences kept within a single split. Accuracy and Macro F1 use the full 3,000-row test split; the shared 256-row board subset checks C/Python prediction parity. No real agricultural accuracy has been measured.

## Key results: v0.3 FP32 / INT8

<!-- V03_MAIN_START -->
| Model | Precision | Test accuracy | Macro F1 | Flash Δ | Raw params | Raw p50 µs | Raw p95 µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | NATIVE | 0.6963 | 0.6604 | 492 B | 20 B | 0.672 ± 0.000 | 0.688 ± 0.000 |
| Tree | NATIVE | 0.8203 | 0.8167 | 752 B | N/A | 0.344 ± 0.000 | 0.359 ± 0.000 |
| Logistic | FP32 | 0.8283 | 0.8256 | 656 B | 208 B | 2.297 ± 0.000 | 2.297 ± 0.000 |
| Logistic | INT8 | 0.8283 | 0.8253 | 624 B | 128 B | 4.375 ± 0.000 | 4.391 ± 0.000 |
| MLP | FP32 | 0.8250 | 0.8214 | 1,396 B | 784 B | 8.359 ± 0.000 | 8.484 ± 0.000 |
| MLP | INT8 | 0.8220 | 0.8178 | 1,104 B | 354 B | 13.750 ± 0.000 | 13.875 ± 0.000 |
<!-- V03_MAIN_END -->

<!-- V03_COMPARE_START -->
| Pair | Δ accuracy | Δ Macro F1 | Flash saving | Raw p50 change |
| --- | ---: | ---: | ---: | ---: |
| Logistic INT8 − FP32 | +0.0000 | -0.0004 | +32 B (+4.9%) | +2.078 µs (+90.5%) |
| Mlp INT8 − FP32 | -0.0030 | -0.0036 | +292 B (+20.9%) | +5.391 µs (+64.5%) |
<!-- V03_COMPARE_END -->

**INT8 saves Flash but is slower in these portable scalar implementations.** Logistic preserves full-test accuracy with slightly lower Macro F1; MLP loses both accuracy and F1. Tree is fastest by raw batch-based latency, while FP32 Logistic has the highest accuracy and Macro F1 on this task. These findings describe this board, workload, and compiler configuration, not all INT8 runtimes.

Flash Δ is compiled flashed-section size relative to the v0.3 no-model harness; raw parameters are exported constants. All six variants have 0 B Static RAM Δ and 0 B inference heap allocation in this build, which does not mean zero RAM use. See [full hardware results](results/v0.3/hardware_benchmark.csv), [v0.3 plots](results/v0.3/plots/), and [integer-core evidence](results/v0.3/integer_core_evidence.txt).

## Implementation and measurement status

- Logistic and MLP have FP32 and INT8 C implementations. Calibration uses only training data; INT8 weights/activations use int32 accumulation and fixed-point hidden-layer requantization. Input StandardScaler preprocessing remains FP32. Optimized ESP-NN/TFLM kernels were not evaluated.
- Physical timing uses ESP32-S3 at 160 MHz, ESP-IDF v5.4.4, performance optimization, and disabled Wi-Fi/Bluetooth. Each variant runs three times, with 256/256 oracle checks before 100 warm-ups and 2,000 batches of 64 predictions.
- The tables report raw per-inference batch averages, including preprocessing and timer overhead. Percentiles describe batch averages, not individual-call tails. Paired timing-baseline and adjusted values remain in the CSV.
- Flash/static RAM come from builds; timing/heap from hardware; accuracy/F1 from synthetic full-test evaluation. Power, energy, and labeled real-sensor accuracy remain unmeasured.

Exact definitions: [quantization methodology](docs/quantization_methodology.md), [measurement methodology](docs/methodology.md), and [hardware protocol](docs/hardware_protocol.md).

## Quick start and reproduction

From the repository root, prepare the host environment and export models:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python training/export_models.py
python training/quantize.py
python benchmark/export_dataset.py
python benchmark/run_benchmark.py
```

To repeat v0.3 hardware measurements, use an ESP-IDF v5.4+ shell, a connected ESP32-S3, and a clean code commit:

```bash
python benchmark/v03_run_hardware.py --port /dev/cu.usbmodemYOURBOARD --runs 3
python benchmark/plot_v03.py
python benchmark/update_readme_v03.py
```

Formal results are saved under `results/v0.3/`; `--allow-dirty` is for exploratory measurements. The [quantization guide](docs/quantization_methodology.md) explains the full workflow. `benchmark/update_readme.py` updates only the archived v0.2/offline blocks; `benchmark/update_readme_v03.py` updates the current comparison.

## Relationship to Smart-Agriculture-Edge-AI

[Smart-Agriculture-Edge-AI](https://github.com/Sver0411/Smart-Agriculture-Edge-AI) uses TinyEdgeBench as a reference for lightweight models and resource evaluation. These isolated benchmark measurements are independent of the main system's firmware and do not establish its board performance.

## Limitations and future work

One synthetic dataset and seed, one board configuration, and scalar FP32/INT8 implementations limit generalization. Real agricultural data, power/energy measurement, and optimized integer kernels are future work. Timer overhead and interrupts limit comparisons at microsecond scale; subset accuracy is separate from full-test accuracy.

## Archived v0.2 evidence

v0.2 FP32 results remain at the [`v0.2` tag](https://github.com/Sver0411/TinyEdgeBench/tree/v0.2) and below. **v0.2 and v0.3 use different sizing harnesses**, so cross-version Flash differences cannot be attributed to quantization. Use the paired v0.3 table above for that comparison.

<details>
<summary>Expand the v0.2 FP32 hardware and offline tables</summary>

<!-- HARDWARE_TABLE_START -->
| Model | Accuracy (3,000) | Macro F1 (3,000) | Flash Δ | Static RAM Δ | Inference heap alloc | p50 µs | p95 µs | p99 µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 40 B | 0 B | 0 B | 0.7 ± 0.0 | 0.7 ± 0.0 | 0.7 ± 0.0 |
| Logistic | 0.8283 | 0.8256 | 204 B | 0 B | 0 B | 2.3 ± 0.0 | 2.3 ± 0.0 | 2.4 ± 0.0 |
| Tree | 0.8203 | 0.8167 | 284 B | 0 B | 0 B | 0.3 ± 0.0 | 0.4 ± 0.0 | 0.4 ± 0.0 |
| MLP | 0.8250 | 0.8214 | 944 B | 0 B | 0 B | 8.4 ± 0.0 | 8.5 ± 0.0 | 8.5 ± 0.0 |
<!-- HARDWARE_TABLE_END -->

Accuracy/F1 use 3,000 synthetic test rows; Flash/static RAM are build measurements and latency/heap are board measurements. The separate 256-row accuracy is in [hardware evidence](results/hardware/).

<!-- OFFLINE_TABLE_START -->
| Model | Accuracy | Macro F1 | Raw constants |
| --- | ---: | ---: | ---: |
| Rule | 0.6963 | 0.6604 | 20 B |
| Logistic | 0.8283 | 0.8256 | 208 B |
| Tree | 0.8203 | 0.8167 | N/A (control flow) |
| MLP | 0.8250 | 0.8214 | 784 B |
<!-- OFFLINE_TABLE_END -->

Raw constants count float32 thresholds, weights, biases, and scaler values; the exported tree is control flow. Full [offline results](results/benchmark.csv), [confusion counts](results/confusion.csv), [accuracy/size plot](results/accuracy_vs_model_size.png), and [confusion plots](results/confusion_matrices.png) remain available.

</details>

## License

MIT — see [LICENSE](LICENSE).
