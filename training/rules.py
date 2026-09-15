"""Rule-based baseline: a handful of hand-written thresholds.

The rules are written from the way the data is generated (see dataset/generate.py),
not fitted to the training set, so they stay a deliberately simple baseline.

Two aggregate features drive everything:

* motion - how much the signal moves over the window (quiet sensor ~ 1.0)
* drift  - how much of that movement has a direction (quiet sensor ~ 0.15)

Their ratio separates "jumping around" (NOISY) from "going somewhere"
(RAPID_CHANGE / SLOW_DRIFT), and the drift size separates fast from slow.

Run:  (used by training/train.py, exported by training/export_models.py)
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.generate import FEATURE_COLUMNS  # noqa: E402

MOTION_INDEX = FEATURE_COLUMNS.index("motion")
DRIFT_INDEX = FEATURE_COLUMNS.index("drift")

# Thresholds. Motion is normalised so that 1.0 = sensor-noise level movement.
MOTION_NOISY = 2.0     # clearly more movement than a quiet sensor
DRIFT_FAST = 0.6       # net movement this large over one window = fast change
DRIFT_ACTIVE = 0.25    # above the ~0.15 a quiet sensor shows by chance
DIRECTED_RATIO = 0.40  # drift/motion: random noise stays well below this
NOISY_RATIO = 0.35     # lots of movement, no direction

# Class ids are shared with the C code (firmware/main/models/model_api.h).
CLASS_NORMAL = 0
CLASS_RAPID_CHANGE = 1
CLASS_SLOW_DRIFT = 2
CLASS_NOISY = 3


def predict_rule(row):
    """Predict one row. `row` is a sequence of 8 floats in FEATURE_COLUMNS order.

    Returns a class id. This function is mirrored line by line in
    firmware/main/models/model_rule.c - keep the two in sync.
    """
    motion = float(row[MOTION_INDEX])
    drift = float(row[DRIFT_INDEX])
    ratio = drift / motion if motion > 1e-6 else 0.0

    if motion >= MOTION_NOISY and ratio < NOISY_RATIO:
        return CLASS_NOISY
    if drift >= DRIFT_FAST and ratio >= DIRECTED_RATIO:
        return CLASS_RAPID_CHANGE
    if drift >= DRIFT_ACTIVE and ratio >= DIRECTED_RATIO:
        return CLASS_SLOW_DRIFT
    if motion >= MOTION_NOISY:
        return CLASS_NOISY
    return CLASS_NORMAL


def predict_rule_batch(rows):
    """Predict a 2-D array / DataFrame of features."""
    return [predict_rule(row) for row in rows]
