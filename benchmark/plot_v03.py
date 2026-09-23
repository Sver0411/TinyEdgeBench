"""Four focused v0.3 figures from the committed CSV evidence."""

from __future__ import annotations

import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v0.3"
PLOTS = OUT / "plots"
ORDER = ("rule", "tree", "logistic_fp32", "logistic_int8", "mlp_fp32", "mlp_int8")


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    offline = {row["variant"]: row for row in read(OUT / "offline_quantization.csv")}
    flash = {row["variant"]: row for row in read(OUT / "flash_size.csv")}
    hardware = read(OUT / "hardware_benchmark.csv")
    latency = {name: statistics.mean(float(row["latency_raw_p50_us"])
                                      for row in hardware if row["variant"] == name)
               for name in ORDER}
    if len(hardware) != 18:
        raise ValueError("need complete v0.3 hardware results")
    PLOTS.mkdir(parents=True, exist_ok=True)

    for metric, title, filename in (("flash", "Accuracy vs compiled Flash Δ", "accuracy_vs_flash.png"),
                                    ("latency", "Accuracy vs raw hardware p50", "accuracy_vs_latency.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in ORDER:
            x = float(flash[name]["flash_delta_bytes"]) if metric == "flash" else latency[name]
            y = float(offline[name]["accuracy"])
            ax.scatter(x, y, s=70)
            ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax.set(xlabel="Compiled Flash Δ (bytes)" if metric == "flash" else "Raw p50 (µs / inference)",
               ylabel="Full-test accuracy", title=title, ylim=(0.62, 0.87))
        ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(PLOTS / filename, dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, key, title in zip(axes, ("accuracy", "flash", "latency"),
                              ("Full-test accuracy", "Compiled Flash Δ (bytes)", "Raw p50 (µs)")):
        families = ("logistic", "mlp")
        fp = [float(offline[f"{f}_fp32"]["accuracy"]) if key == "accuracy" else
              float(flash[f"{f}_fp32"]["flash_delta_bytes"]) if key == "flash" else latency[f"{f}_fp32"]
              for f in families]
        q = [float(offline[f"{f}_int8"]["accuracy"]) if key == "accuracy" else
             float(flash[f"{f}_int8"]["flash_delta_bytes"]) if key == "flash" else latency[f"{f}_int8"]
             for f in families]
        ax.bar([-.18, .82], fp, width=.36, label="FP32")
        ax.bar([.18, 1.18], q, width=.36, label="INT8")
        ax.set_xticks([0, 1], ["Logistic", "MLP"]); ax.set_title(title)
    axes[0].legend(); fig.tight_layout(); fig.savefig(PLOTS / "fp32_vs_int8.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for family, color in (("logistic", "tab:blue"), ("mlp", "tab:orange")):
        fp, q = f"{family}_fp32", f"{family}_int8"
        loss = float(offline[fp]["accuracy"]) - float(offline[q]["accuracy"])
        saving = int(flash[fp]["flash_delta_bytes"]) - int(flash[q]["flash_delta_bytes"])
        ax.scatter(saving, loss, s=110, color=color)
        ax.annotate(family, (saving, loss), xytext=(7, 4), textcoords="offset points")
    ax.axhline(0, color="grey", linewidth=.8)
    ax.set(xlabel="Compiled Flash saving (bytes)", ylabel="Full-test accuracy loss",
           title="Quantization trade-off")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(PLOTS / "accuracy_loss_vs_flash_saving.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
