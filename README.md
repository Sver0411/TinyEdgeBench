# TinyEdgeBench

TinyEdgeBench compares several lightweight classification methods on the same sensor-data task, to see how much accuracy a more complex model actually buys and how much extra memory that complexity costs.

## What it does

One small task, four methods of increasing complexity, one shared dataset:

* the task is to label the current state of a sensor stream as `NORMAL`, `RAPID_CHANGE`, `SLOW_DRIFT` or `NOISY`
* the methods are a hand-written rule set, logistic regression, a small decision tree, and a tiny MLP
* every method gets the same 8 features and the same train / validation / test rows
* each method is scored for accuracy and macro F1, sized by its raw parameter bytes, and exported as plain C that an ESP32 can compile

## How it works

```text
dataset/generate.py     synthetic sensor sequences -> 20,000 rows, 8 features, 4 classes
training/train.py       train logistic / tree / mlp, score all four methods
benchmark/run_benchmark.py   main table + 2 plots into results/
training/export_models.py    write firmware/main/models/model_*.c (float32 inference)
firmware/               minimal ESP-IDF project that runs all four predictors
```

## Models

| Method | What it is | Settings |
| --- | --- | --- |
| Rule-based | 5 hand-written thresholds on `motion` and `drift` | not fitted to the data |
| Logistic Regression | `StandardScaler` + `LogisticRegression` | max_iter=1000 |
| Decision Tree | `DecisionTreeClassifier` | max_depth=5 |
| Tiny MLP | `StandardScaler` + `MLPClassifier` | 8 -> 8 -> 4, ReLU, max_iter=500 |

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

| Method              | Accuracy | Macro F1 | Parameters | Est. raw size |
| ------------------- | -------: | -------: | ---------: | ------------: |
| Rule-based          |   0.6963 |   0.6604 |          5 |         20 B  |
| Logistic Regression |   0.8283 |   0.8256 |         52 |        208 B  |
| Decision Tree       |   0.8203 |   0.8167 |         57 |        281 B  |
| Tiny MLP            |   0.8250 |   0.8214 |        196 |        784 B  |

Reading the table:

* the hand-written rules are 10x smaller than anything else but give up ~13 accuracy points
* logistic regression is both the smallest *and* the most accurate learned method here - the class boundaries are mostly linear in `(motion, drift)`
* the tree and the MLP are within noise of logistic regression (0.82 +- 0.01), while the MLP stores ~4x the parameters
* most remaining errors are `RAPID_CHANGE` confused with `SLOW_DRIFT`: inside an 8-sample window a small fast step and a steady drift look alike

Size is *estimated raw parameter storage* (float32 constants the inference code needs, plus the scaler where one is used), not pickle size and not compiled flash footprint. See the Limitations below.

Plots: `results/accuracy_vs_model_size.png`, `results/confusion_matrices.png`.

## ESP32 status

* `idf.py set-target esp32s3 && idf.py build` succeeds with ESP-IDF v5.4.4 (total image size 192,552 bytes)
* `firmware/main/main.c` runs all four predictors on four fixed rows and prints the results
* `measure_latency()` in `main.c` is the `esp_timer` hook for on-board timing

```text
Hardware latency: Not measured yet
Per-model flash / RAM: not separated (would need one build per model)
```

Nothing has been run on a board, so no latency, power or RAM numbers are published.

## Run it

```bash
python dataset/generate.py          # writes dataset/data.csv
python training/train.py            # trains, prints val/test scores
python training/export_models.py    # writes firmware/main/models/model_*.c
python benchmark/run_benchmark.py   # writes results/benchmark.csv + 2 plots
python -m pytest tests/ -v          # 26 tests, incl. Python/C parity
```

Firmware (requires ESP-IDF v5.4+):

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
```

## Project structure

```text
TinyEdgeBench/
├── dataset/
│   ├── generate.py
│   └── data.csv
├── training/
│   ├── rules.py            hand-written rule baseline
│   ├── train.py            train + evaluate + size
│   └── export_models.py    Python -> C
├── benchmark/
│   └── run_benchmark.py
├── firmware/
│   ├── CMakeLists.txt
│   └── main/
│       ├── CMakeLists.txt
│       ├── main.c
│       └── models/         model_api.h + 4 generated model_*.c
├── results/
│   ├── benchmark.csv
│   ├── confusion.csv
│   ├── accuracy_vs_model_size.png
│   └── confusion_matrices.png
├── tests/
│   ├── conftest.py
│   ├── parity_host.c
│   ├── test_dataset.py
│   ├── test_models.py
│   └── test_export.py
├── README.md
├── requirements.txt
└── LICENSE
```

Python/C parity: `tests/parity_host.c` compiles the generated C on the host and checks 100 test samples against the Python predictions.

## Limitations

* the data is synthetic, so these numbers say nothing yet about real sensors
* nothing has been measured on hardware - no latency, no power, no real RAM profile
* "model size" is estimated raw parameter bytes, not the flash a compiler actually emits, and it ignores interpreter/runtime code
* float32 only, no quantisation
* one task, one dataset size, one seed - a single data point, not a general law

## Future work

* Real ESP32-S3 latency
* Real RAM profiling
* INT8 quantisation
* Real sensor traces

## License

MIT
