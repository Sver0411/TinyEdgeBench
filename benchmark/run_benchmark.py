"""Run the PC benchmark.

Trains all four methods on the shared split, scores them on the shared test
set, estimates raw parameter storage, and writes:

    results/benchmark.csv          main table
    results/confusion.csv          confusion matrices (long format)
    results/accuracy_vs_model_size.png
    results/confusion_matrices.png

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


def run(methods=None):
    """Train, evaluate and size every method. Returns a list of result dicts."""
    splits = train.load_data()
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    models = train.train_models(X_train, y_train)
    methods = methods or train.METHODS

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
            "parameters": size["parameters"],
            "size_bytes": size["bytes"],
            "size_detail": size["detail"],
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
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    # alternate the label offsets so neighbouring points do not overlap
    offsets = [((10, -4), "left"), ((10, -14), "left"), ((10, 8), "left"), ((10, -4), "left")]
    for row, (offset, align) in zip(results, offsets):
        ax.scatter(row["size_bytes"], row["accuracy"], s=90, zorder=3)
        ax.annotate(row["method"], (row["size_bytes"], row["accuracy"]),
                    textcoords="offset points", xytext=offset, ha=align)
    sizes = [row["size_bytes"] for row in results]
    ax.set_xlim(min(sizes) * 0.4, max(sizes) * 2.5)
    ax.set_xscale("log")
    ax.set_xlabel("estimated raw parameter storage (bytes, log scale)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs model size")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    plot_accuracy_vs_size(results, RESULTS_DIR / "accuracy_vs_model_size.png")
    plot_confusion_matrices(matrices, RESULTS_DIR / "confusion_matrices.png")

    print(f"{'method':10s} {'acc':>7s} {'macroF1':>9s} {'params':>8s} {'size':>8s}")
    for row in results:
        print(f"{row['method']:10s} {row['accuracy']:7.4f} {row['macro_f1']:9.4f} "
              f"{row['parameters']:8d} {row['size_bytes']:8d} B")
    print(f"\nwrote {RESULTS_DIR / 'benchmark.csv'}")
    print(f"wrote {RESULTS_DIR / 'confusion.csv'}")
    print(f"wrote {RESULTS_DIR / 'accuracy_vs_model_size.png'}")
    print(f"wrote {RESULTS_DIR / 'confusion_matrices.png'}")


if __name__ == "__main__":
    main()
