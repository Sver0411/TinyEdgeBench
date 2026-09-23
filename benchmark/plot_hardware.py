"""Plot only validated, parsed hardware runs from the result CSV."""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/hardware/hardware_benchmark.csv"
OUT = ROOT / "results/plots"


def main():
    data = pd.read_csv(CSV)
    expected = ("rule", "logistic", "tree", "mlp")
    if set(data.model) != set(expected) or data.groupby("model").size().min() < 3:
        raise SystemExit("need at least three validated hardware runs for all four models")
    avg = data.groupby("model", sort=False).mean(numeric_only=True).loc[list(expected)]
    with (ROOT / "results/benchmark.csv").open(newline="") as handle:
        quality = {row["method"]: float(row["accuracy"]) for row in csv.DictReader(handle)}
    OUT.mkdir(parents=True, exist_ok=True)

    for xcol, xlabel, name in (
        ("flash_delta_bytes", "Compiled Flash delta (bytes)", "accuracy_vs_flash.png"),
        ("latency_p50_us", "ESP32-S3 median inference latency (µs)", "accuracy_vs_latency.png"),
    ):
        if avg[xcol].isna().any():
            raise SystemExit(f"missing {xcol}; run benchmark/measure_flash.py first")
        fig, ax = plt.subplots(figsize=(6, 4))
        for model, row in avg.iterrows():
            ax.scatter(row[xcol], quality[model], s=80)
            offset = {"logistic": (8, -18), "tree": (8, 7)}.get(model, (8, 5))
            ax.annotate(model, (row[xcol], quality[model]), xytext=offset,
                        textcoords="offset points")
        ax.set(xlabel=xlabel, ylabel="Accuracy on full 3,000-row test split", ylim=(0, 1))
        ax.set_xlim(max(0, avg[xcol].min() * .75), avg[xcol].max() * 1.18)
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=150)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, col, title in zip(axes,
                              ("flash_delta_bytes", "static_ram_delta_bytes", "latency_p50_us"),
                              ("Flash Δ (bytes)", "Static RAM Δ (bytes)", "Median latency (µs)")):
        if (avg[col] == 0).all():
            ax.text(.5, .5, "0 B for all models", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xticks([])
        else:
            ax.bar(avg.index, avg[col])
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(OUT / "resource_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
