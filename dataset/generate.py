"""Synthetic sensor dataset generator for TinyEdgeBench.

Short sequences of temperature / humidity / light readings are generated for
four environment states, then converted into flat rows with a small fixed
feature set. The data is synthetic - see README "Dataset".

Run:
    python dataset/generate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "dataset" / "data.csv"

SEED = 42
N_ROWS = 20000
ROWS_PER_SEQUENCE = 10
WINDOW = 8                      # samples used by the motion / drift features
SEQ_LEN = WINDOW + ROWS_PER_SEQUENCE

LABELS = ("NORMAL", "RAPID_CHANGE", "SLOW_DRIFT", "NOISY")
CHANNELS = ("temp", "hum", "light")

# Noise of a quiet sensor. Also used to normalise the motion / drift features,
# so "motion = 1.0" means "moving about as much as sensor noise".
REF_SIGMA = {"temp": 0.12, "hum": 0.6, "light": 6.0}
BASELINE = {"temp": (24.0, 2.0), "hum": (50.0, 8.0), "light": (300.0, 90.0)}

# Per-class generation ranges (see README "Dataset" for what each class means).
NOISY_MULTIPLIER = (3.0, 8.0)    # noise sigma multiplier, applied to 1-3 channels
DRIFT_RATE = (0.15, 0.8)         # REF_SIGMA units per sample, 1-3 channels
RAPID_MAGNITUDE = (4.0, 30.0)    # total step size in REF_SIGMA units, 1-2 channels
RAPID_DURATION = (3, 4)          # samples the step takes
RAPID_START = (7, 8)             # first sample of the step

FEATURE_COLUMNS = [
    "temp", "hum", "light",
    "d_temp", "d_hum", "d_light",
    "motion", "drift",
]
SPLIT_FRACTIONS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def _pick_channels(rng, n_max):
    """Pick 1..n_max channels that behave unusually."""
    n = int(rng.integers(1, n_max + 1))
    return rng.choice(CHANNELS, size=n, replace=False)


def _make_levels(rng, label):
    """Return {channel: level array} for one synthetic sequence."""
    t = np.arange(SEQ_LEN, dtype=float)
    bases, levels = {}, {}
    for ch in CHANNELS:
        mu, sd = BASELINE[ch]
        bases[ch] = float(rng.normal(mu, sd))
        levels[ch] = bases[ch] + rng.normal(0.0, REF_SIGMA[ch], SEQ_LEN)

    if label == "NOISY":
        for ch in _pick_channels(rng, len(CHANNELS)):
            m = rng.uniform(*NOISY_MULTIPLIER)
            levels[ch] = bases[ch] + rng.normal(0.0, REF_SIGMA[ch] * m, SEQ_LEN)

    elif label == "SLOW_DRIFT":
        for ch in _pick_channels(rng, len(CHANNELS)):
            rate = rng.uniform(*DRIFT_RATE) * REF_SIGMA[ch]
            levels[ch] = levels[ch] + rate * float(rng.choice([-1.0, 1.0])) * t

    elif label == "RAPID_CHANGE":
        for ch in _pick_channels(rng, 2):
            amp = rng.uniform(*RAPID_MAGNITUDE) * REF_SIGMA[ch]
            amp *= float(rng.choice([-1.0, 1.0]))
            dur = int(rng.integers(RAPID_DURATION[0], RAPID_DURATION[1] + 1))
            start = int(rng.integers(RAPID_START[0], RAPID_START[1] + 1))
            step = np.clip((t - start + 1.0) / dur, 0.0, 1.0)
            levels[ch] = levels[ch] + amp * step

    return levels


def _sequence_rows(levels):
    """Turn one sequence of levels into ROWS_PER_SEQUENCE feature rows."""
    idx = np.arange(WINDOW, SEQ_LEN)
    deltas, motion_c, drift_c = {}, [], []

    for ch in CHANNELS:
        level = levels[ch]
        d = np.empty(SEQ_LEN)
        d[0] = 0.0
        d[1:] = level[1:] - level[:-1]
        deltas[ch] = d

        csum = np.concatenate([[0.0], np.cumsum(d)])
        cabs = np.concatenate([[0.0], np.cumsum(np.abs(d))])
        win_sum = csum[idx + 1] - csum[idx + 1 - WINDOW]
        win_abs = cabs[idx + 1] - cabs[idx + 1 - WINDOW]

        motion_c.append(win_abs / WINDOW / REF_SIGMA[ch])
        drift_c.append(np.abs(win_sum) / WINDOW / REF_SIGMA[ch])

    motion = np.mean(np.vstack(motion_c), axis=0)
    drift = np.mean(np.vstack(drift_c), axis=0)

    rows = {
        "temp": levels["temp"][idx],
        "hum": levels["hum"][idx],
        "light": levels["light"][idx],
        "d_temp": deltas["temp"][idx],
        "d_hum": deltas["hum"][idx],
        "d_light": deltas["light"][idx],
        "motion": motion,
        "drift": drift,
    }
    return rows


def build_dataset():
    """Build the full dataset as a DataFrame (deterministic for a fixed seed)."""
    rng = np.random.default_rng(SEED)
    n_sequences = N_ROWS // ROWS_PER_SEQUENCE

    # Exact class balance, then shuffle.
    labels = np.array(list(LABELS) * (n_sequences // len(LABELS)))
    labels = rng.permutation(labels)

    # Splits are assigned per sequence so rows of one sequence never straddle.
    n_train = int(round(n_sequences * SPLIT_FRACTIONS[0][1]))
    n_val = int(round(n_sequences * SPLIT_FRACTIONS[1][1]))
    splits = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * (n_sequences - n_train - n_val)
    )
    splits = rng.permutation(splits)

    frames = []
    for seq_id, (label, split) in enumerate(zip(labels, splits)):
        rows = _sequence_rows(_make_levels(rng, label))
        frame = pd.DataFrame(rows)
        frame.insert(0, "seq_id", seq_id)
        frame.insert(1, "split", split)
        frame["label"] = label
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data[["seq_id", "split", "label"] + FEATURE_COLUMNS]
    return data.round(6)


def main():
    data = build_dataset()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)

    counts = data["split"].value_counts()
    print(f"wrote {OUT_PATH} ({len(data)} rows, {len(data.columns)} columns)")
    for split, _ in SPLIT_FRACTIONS:
        print(f"  {split:5s} {int(counts[split]):6d} rows")
    for label in LABELS:
        print(f"  {label:13s} {int((data['label'] == label).sum()):6d} rows")


if __name__ == "__main__":
    main()
