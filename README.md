# TinyEdgeBench

**English** | [简体中文](README.zh-CN.md)

TinyEdgeBench compares several lightweight classification methods on the same sensor-data task, to see how much accuracy a more complex model actually buys and how much extra memory that complexity costs.

## What it does

One small task, four methods of increasing complexity, one shared dataset:

* the task is to classify the recent behaviour of a short sensor window (the last 8 samples) as `NORMAL`, `RAPID_CHANGE`, `SLOW_DRIFT` or `NOISY`
* the methods are a hand-written rule set, logistic regression, a small decision tree, and a tiny MLP
* every method gets the same 8 features and the same train / validation / test rows
* each method is scored for accuracy and macro F1, sized in two different ways, and exported as plain C that an ESP32 can compile

## How it works

```text
dataset/generate.py          synthetic sensor sequences -> 20,000 rows, 8 features, 4 classes
training/train.py            train logistic / tree / mlp, score all four methods
benchmark/run_benchmark.py   main table + 2 plots into results/
benchmark/measure_flash.py   build one predictor at a time, read real ELF sizes
training/export_models.py    write firmware/main/models/model_*.c (float32 inference)
firmware/                    minimal ESP-IDF project that runs the linked predictors
```

## Models

| Method | What it is | Settings |
| --- | --- | --- |
| Rule-based | 5 hand-written thresholds on `motion` and `drift` | not fitted to the data |
| Logistic Regression | `StandardScaler` + `LogisticRegression` | max_iter=1000 |
| Decision Tree | `DecisionTreeClassifier` | max_depth=5 |
| Tiny MLP | `StandardScaler` + `MLPClassifier` | 8 -> 8 -> 8 -> 4, ReLU, max_iter=500 |

No TensorFlow, no ONNX, no quantisation - v0.1 compares the methods first.

## Dataset

Synthetic. The generator (`dataset/generate.py`, seed 42) creates short sequences of temperature / humidity / light readings and turns them into rows.

```text
rows       20,000 (2,000 sequences x 10 rows)
features   8  - temp, hum, light, d_temp, d_hum, d_light, motion, drift
classes    4  - NORMAL, RAPID_CHANGE, SLOW_DRIFT, NOISY (5,000 rows each)
split      70 / 15 / 15, assigned per sequence so one sequence never straddles splits
```

* `NORMAL` - baseline plus sensor noise only
* `RAPID_CHANGE` - one or two channels step to a new level within 3-4 samples
* `SLOW_DRIFT` - one to three channels keep moving slowly in one direction
* `NOISY` - one to three channels get 3x-8x the normal noise, with no direction

`motion` and `drift` are computed over an 8-sample window and normalised so that 1.0 means "moves about as much as sensor noise". The classes overlap on purpose: a small fast step can look like steady drift, and a quiet sensor with a little drift can look normal. No single feature reaches 90% accuracy on its own (best: `motion`, ~61%).

> Synthetic data is used for the initial benchmark. Real ESP32 sensor traces can be added later.

## Results

Test split, 3,000 rows, seed 42:

| Method              | Accuracy | Macro F1 | Raw constants    | Compiled Flash Δ |
| ------------------- | -------: | -------: | :--------------- | ---------------: |
| Rule-based          |   0.6963 |   0.6604 | 20 B             |           968 B  |
| Logistic Regression |   0.8283 |   0.8256 | 208 B            |         1,168 B  |
| Decision Tree       |   0.8203 |   0.8167 | N/A (structural) |         1,308 B  |
| Tiny MLP            |   0.8250 |   0.8214 | 784 B            |         1,904 B  |

Reading the table:

* the hand-written rules are clearly the weakest but also the cheapest by both measures
* logistic regression is both the most accurate method here and cheaper than the tree or the MLP - the class boundaries are mostly linear in `(motion, drift)`
* the tree and the MLP land within noise of logistic regression (0.82 +- 0.01) while costing more flash
* the decision tree has no "raw constants" number: it is exported as nested `if/else` comparisons, so there is no parameter array whose byte count would mean anything
* most remaining errors are `RAPID_CHANGE` confused with `SLOW_DRIFT`: inside an 8-sample window a small fast step and a steady drift look alike

### The two size columns mean different things

**Raw constants** are the theoretical data a method has to store: weights, biases, scaler constants and thresholds, counted as float32 values. They describe how much the *model* carries, not how it compiles, and they are not even defined the same way for every method:

| Method | Raw constants | Why |
| --- | --- | --- |
| Rule-based | 20 B | 5 thresholds |
| Logistic Regression | 208 B | 32 weights + 4 biases + 16 scaler |
| Decision Tree | N/A | no constants - exported as nested comparisons (control flow) |
| Tiny MLP | 784 B | 180 weights/biases + 16 scaler |

Do not put the tree's row next to the others: it has no parameter array at all. `results/benchmark.csv` therefore reports it as empty in `raw_constants_bytes` and keeps the rough node-and-leaf count only in a separate `structural_estimate_bytes` column.

**Compiled Flash Δ** is measured, not estimated. `benchmark/measure_flash.py` builds the firmware five times with the CMake option `-DTINYEDGEBENCH_MODEL=...` - once with no predictor (baseline) and once per method - and reads the ELF section sizes from the ESP-IDF toolchain:

```text
flash delta = flash bytes(model build) - flash bytes(baseline build)

baseline (no predictor) = 189,492 bytes of flash image
```

This is a compiled flash-section delta: it covers the predictor **plus** the small fixed amount of demo / scaffolding code that any non-empty build pulls in (the baseline stubs those functions out). That makes it comparable across the four methods, but it should not be read as a pure model size.

**When comparing the four C implementations, use Compiled Flash Δ** - including in `accuracy_vs_model_size.png`, whose x axis is the compiled flash delta from `results/flash_size.csv`, not the raw-constant count.

Plots: `results/accuracy_vs_model_size.png` (accuracy vs compiled flash), `results/confusion_matrices.png`. Raw numbers: `results/benchmark.csv`, `results/flash_size.csv`.

## ESP32 status

| Quantity | Status |
| --- | --- |
| Firmware build | OK - ESP-IDF v5.4.4, `idf.py set-target esp32s3 && idf.py build` |
| Compiled flash | Build-time measured (no hardware needed) |
| Hardware latency | Not measured yet |
| Real RAM | Not measured yet |
| Power | Not measured yet |
| Per-model RAM | Not separated (would need one runtime profile per model) |

`firmware/main/main.c` runs the linked predictors on four fixed rows and prints the results. `measure_latency()` is the `esp_timer` hook kept ready for when a board exists - until then no latency, RAM or power numbers are published.

Build-time flash per predictor (needs an ESP-IDF environment: source its
`export.sh`, or set `IDF_PATH`):

```bash
python benchmark/measure_flash.py     # -> results/flash_size.csv
```

The scripts look ESP-IDF up in this order: `$IDF_PATH/export.sh`, then `idf.py`
already on `PATH`. If neither exists they print

```text
ESP-IDF environment not found.
Run ESP-IDF export.sh first or set IDF_PATH.
```

## Run it

```bash
python dataset/generate.py          # writes dataset/data.csv
python training/train.py            # trains, prints val/test scores
python training/export_models.py    # writes firmware/main/models/model_*.c
python benchmark/run_benchmark.py   # writes results/benchmark.csv + 2 plots
python benchmark/measure_flash.py   # writes results/flash_size.csv (needs ESP-IDF env)
python -m pytest tests/ -v          # includes the Python/C parity check
```

Firmware (requires ESP-IDF v5.4+):

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
```

Reproducibility: everything is seeded, and no pickles are stored - the export and the benchmark retrain from the same fixed seed.

Results in this repository were generated with:

```text
Python 3.13.12
NumPy 2.5.3
pandas 3.0.5
scikit-learn 1.9.1
matplotlib 3.11.2
ESP-IDF v5.4.4 (target esp32s3)
```

## Project structure

```text
TinyEdgeBench/
├── dataset/
│   ├── generate.py
│   └── data.csv
├── training/
│   ├── rules.py            hand-written rule baseline
│   ├── train.py            train + evaluate + raw-constant size
│   └── export_models.py    Python -> C
├── benchmark/
│   ├── run_benchmark.py    accuracy / macro F1 / plots
│   └── measure_flash.py    compiled flash per predictor
├── firmware/
│   ├── CMakeLists.txt
│   └── main/
│       ├── CMakeLists.txt  TINYEDGEBENCH_MODEL selects what gets linked
│       ├── main.c
│       └── models/         model_api.h + 4 generated model_*.c
├── results/
│   ├── benchmark.csv
│   ├── confusion.csv
│   ├── flash_size.csv
│   ├── accuracy_vs_model_size.png
│   └── confusion_matrices.png
├── tests/
│   ├── conftest.py
│   ├── parity_host.c
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_export.py
├── .github/workflows/tests.yml   checkout + install + pytest
├── README.md
├── README.zh-CN.md
├── requirements.txt
└── LICENSE
```

Python/C parity: `tests/parity_host.c` compiles the generated C on the host and checks 100 test samples against the Python predictions.

## Limitations

* the data is synthetic, so these numbers say nothing yet about real sensors
* nothing has been measured on hardware - no latency, no power, no RAM profile
* compiled flash Δ includes the demo scaffolding, so it is comparable between methods but not a pure "model" size
* float32 only, no quantisation
* one task, one dataset size, one seed - a single data point, not a general law

## Future work

* Real ESP32-S3 latency
* Real RAM profiling
* INT8 quantisation
* Real sensor traces

## License

MIT
