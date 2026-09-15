"""Run the PC benchmark.

Trains all four methods on the shared split, scores them on the shared test
set, records the raw-constant bookkeeping, and writes:

    results/benchmark.csv          main table (metrics + both size columns)
    results/confusion.csv          confusion matrices (long format)
    results/accuracy_vs_model_size.png   accuracy vs compiled flash delta
    results/confusion_matrices.png

The compiled flash delta comes from results/flash_size.csv, so run
benchmark/measure_flash.py (needs ESP-IDF) first if you want the plot.

Run:
    python benchmark/run_benchmark.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.generate import LABELS  # noqa: E402
from training import train  # noqa: E402

RESULTS_DIR = ROOT / "results"


def load_flash_delta():
    """Read {method: compiled flash delta} from results/flash_size.csv.

    Returns {} when the file does not exist yet - run benchmark/measure_flash.py
    (needs ESP-IDF) to produce it. Delta values are only available for methods
    that have a C implementation.
    """
    path = RESULTS_DIR / "flash_size.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "flash_delta_bytes" not in frame.columns:
        return {}
    return dict(zip(frame["method"], frame["flash_delta_bytes"]))


def run(methods=None):
    """Train, evaluate and size every method. Returns rows and confusion rows."""
    splits = train.load_data()
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    models = train.train_models(X_train, y_train)
    methods = methods or train.METHODS
    flash_delta = load_flash_delta()

    rows, confusion_rows = [], []
    for name in methods:
        metrics = train.evaluate(y_test, train.predict_ids(name, models[name], X_test))
        size = train.estimate_size(name, models[name])
        rows.append({
            "method": name,
            "accuracy": round(metrics["accuracy"], 4),
            "macro_precision": round(metrics["macro_precision"], 4),
            "macro_recall": round(metrics["macro_recall"], 4),
            "macro_f1": round(metrics["macro_f1"], 4),
            # empty string instead of NaN keeps the CSV readable
            "raw_constants_bytes": size["raw_constants_bytes"] if size["raw_constants_bytes"] is not None else "",
            "raw_constants_note": size["raw_constants_note"],
            "structural_estimate_bytes": size["structural_estimate_bytes"] if size["structural_estimate_bytes"] is not None else "",
            "compiled_flash_delta_bytes": flash_delta.get(name, ""),
        })
        for i, true_label in enumerate(LABELS):
            for j, pred_label in enumerate(LABELS):
                confusion_rows.append({
                    "method": name,
                    "true": true_label,
                    "predicted": pred_label,
                    "count": int(metrics["confusion"][i, j]),
                })
    return rows, confusion_rows


def plot_accuracy_vs_size(results, path):
    """Accuracy against compiled flash delta (see benchmark/measure_flash.py)."""
    plotted = [row for row in results if row["compiled_flash_delta_bytes"] != ""]
    if not plotted:
        print("skipping accuracy plot: results/flash_size.csv is missing "
              "- run python benchmark/measure_flash.py first")
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # alternate the label offsets so neighbouring points do not overlap
    offsets = [((10, -4), "left"), ((10, -14), "left"), ((10, 8), "left"), ((10, -4), "left")]
    sizes = [int(row["compiled_flash_delta_bytes"]) for row in plotted]
    for row, (offset, align) in zip(plotted, offsets):
        size = int(row["compiled_flash_delta_bytes"])
        ax.scatter(size, row["accuracy"], s=90, zorder=3)
        ax.annotate(row["method"], (size, row["accuracy"]),
                    textcoords="offset points", xytext=offset, ha=align)
    ax.set_xlim(min(sizes) * 0.85, max(sizes) * 1.15)
    ax.set_xlabel("compiled flash delta vs baseline (bytes)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs Compiled Flash")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrices(confusion, path):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    for ax, row in zip(axes.ravel(), confusion):
        matrix = np.array(row["matrix"])
        ax.imshow(matrix, cmap="Blues")
        ax.set_title(row["method"])
        ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.suptitle("Confusion matrices (test split)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results, confusion_rows = run()

    table = pd.DataFrame(results)
    table.to_csv(RESULTS_DIR / "benchmark.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(RESULTS_DIR / "confusion.csv", index=False)

    matrices = []
    for row in results:
        frame = pd.DataFrame(confusion_rows)
        subset = frame[frame["method"] == row["method"]]
        matrices.append({
            "method": row["method"],
            "matrix": subset.pivot(index="true", columns="predicted", values="count")
                            .reindex(index=LABELS, columns=LABELS).to_numpy(),
        })
    accuracy_plot = plot_accuracy_vs_size(results, RESULTS_DIR / "accuracy_vs_model_size.png")
    plot_confusion_matrices(matrices, RESULTS_DIR / "confusion_matrices.png")

    print(f"{'method':10s} {'acc':>7s} {'macroF1':>9s} {'rawConst':>10s} {'flashDelta':>12s}")
    for row in results:
        raw = "N/A" if row["raw_constants_bytes"] == "" else f"{int(row['raw_constants_bytes'])} B"
        flash = row["compiled_flash_delta_bytes"]
        flash = "n/a" if flash == "" else f"{int(flash)} B"
        print(f"{row['method']:10s} {row['accuracy']:7.4f} {row['macro_f1']:9.4f} "
              f"{raw:>10s} {flash:>12s}")
    print(f"\nwrote {RESULTS_DIR / 'benchmark.csv'}")
    print(f"wrote {RESULTS_DIR / 'confusion.csv'}")
    if accuracy_plot is not None:
        print(f"wrote {accuracy_plot}")
    print(f"wrote {RESULTS_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
