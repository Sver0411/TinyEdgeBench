"""Train-only PTQ, exact Python scalar-INT8 reference, and C artifact export.

The fixed per-tensor scheme is selected before test evaluation. Train supplies
all calibration ranges; validation may assess the scheme; test is only scored.
Run ``python training/quantize.py`` to refresh the v0.3 generated artifacts.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.export_dataset import selected_rows  # noqa: E402
from dataset.generate import FEATURE_COLUMNS, LABELS  # noqa: E402
from training import train  # noqa: E402

OUT = ROOT / "results/v0.3"
MODELS = ROOT / "firmware/main/models"
EXPECTED = ROOT / "firmware/main/benchmark_int8_expected.c"
SHIFT = 24
LIMIT = 127


def round_half_away(values):
    values = np.asarray(values, dtype=np.float64)
    return (np.sign(values) * np.floor(np.abs(values) + 0.5)).astype(np.int64)


def scale(values):
    maximum = float(np.max(np.abs(values)))
    if maximum <= 0:
        raise ValueError("calibration range is zero")
    return float(np.float32(maximum / LIMIT))


def quantize(values, quant_scale):
    return np.clip(round_half_away(np.asarray(values, dtype=np.float64) / quant_scale),
                   -LIMIT, LIMIT).astype(np.int8)


def scaler_constants(model):
    scaler = model.named_steps["scaler"]
    return (scaler.mean_.astype(np.float32),
            (1.0 / scaler.scale_).astype(np.float32))


def standardized(features, params):
    x = np.asarray(features, dtype=np.float32)
    mean = np.asarray(params["scaler_mean"], dtype=np.float32)
    inverse = np.asarray(params["scaler_inv_std"], dtype=np.float32)
    return ((x - mean) * inverse).astype(np.float32)


def quantized_input(features, params):
    x = standardized(features, params)
    # Matches two float32 operations in the generated C preprocessing.
    pre = (x * np.float32(params["input_inv_scale"])).astype(np.float32)
    clipped = np.count_nonzero(np.abs(pre) > LIMIT)
    q = np.clip(round_half_away(pre), -LIMIT, LIMIT).astype(np.int8)
    return q, int(clipped), int(pre.size)


def fixed_multiplier(input_scale, weight_scale, output_scale):
    ratio = input_scale * weight_scale / output_scale
    multiplier = int(round_half_away(ratio * (1 << SHIFT)))
    if multiplier <= 0 or multiplier > np.iinfo(np.int32).max:
        raise ValueError("fixed-point multiplier outside int32")
    return multiplier


def requantize_relu(accumulator, multiplier):
    product = np.asarray(accumulator, dtype=np.int64) * int(multiplier)
    half = 1 << (SHIFT - 1)
    rounded = np.sign(product) * ((np.abs(product) + half) // (1 << SHIFT))
    clipped = int(np.count_nonzero(rounded > LIMIT))
    return np.clip(rounded, 0, LIMIT).astype(np.int8), clipped, int(rounded.size)


def _matrix(model, layer):
    return model.named_steps["clf"].coefs_[layer].T.astype(np.float32)


def _bias(model, layer):
    return model.named_steps["clf"].intercepts_[layer].astype(np.float32)


def _quantized_layer(weights, bias, input_scale, output_scale=None):
    weight_scale = scale(weights)
    qweights = quantize(weights, weight_scale)
    bias_scale = input_scale * weight_scale
    qbias64 = round_half_away(np.asarray(bias, np.float64) / bias_scale)
    worst = int(np.max(np.abs(qbias64))) + weights.shape[1] * LIMIT * LIMIT
    if worst > np.iinfo(np.int32).max:
        raise ValueError("int32 accumulator can overflow")
    data = {
        "weight_scale": weight_scale,
        "bias_scale": float(bias_scale),
        "weights": qweights.tolist(),
        "bias": qbias64.astype(np.int32).tolist(),
        "accumulator_absolute_bound": worst,
    }
    if output_scale is not None:
        multiplier = fixed_multiplier(input_scale, weight_scale, output_scale)
        if worst * multiplier + (1 << (SHIFT - 1)) > np.iinfo(np.int64).max:
            raise ValueError("int64 requantization can overflow")
        data.update(output_scale=output_scale, multiplier=multiplier, shift=SHIFT,
                    requantization_product_bound=worst * multiplier)
    return data


def calibrate(models, x_train):
    result = {
        "scheme": "symmetric_signed_int8_per_tensor_ptq",
        "calibration_split": "train",
        "selection_split": "none_fixed_before_test",
        "test_used_for_calibration": False,
        "rounding": "nearest_ties_away_from_zero",
        "saturation": [-LIMIT, LIMIT],
        "requantization": "signed_half_away_integer_multiplier_shift_24",
    }
    for family in ("logistic", "mlp"):
        model = models[family]
        mean, inverse = scaler_constants(model)
        common = {"scaler_mean": mean.tolist(), "scaler_inv_std": inverse.tolist()}
        scaled_train = standardized(x_train, common)
        s0 = scale(scaled_train)
        common.update(input_scale=s0, input_inv_scale=float(np.float32(1.0 / s0)),
                      input_calibration_max_abs=float(np.max(np.abs(scaled_train))))
        if family == "logistic":
            clf = model.named_steps["clf"]
            layer = _quantized_layer(clf.coef_.astype(np.float32),
                                     clf.intercept_.astype(np.float32), s0)
            common.update(layer=layer, raw_parameter_bytes=64 + 4 + 4 + 4 + 4 + 32 + 16,
                          declared_local_scratch_bytes=12)
        else:
            w1, w2, w3 = (_matrix(model, i) for i in range(3))
            b1, b2, b3 = (_bias(model, i) for i in range(3))
            h1 = np.maximum(scaled_train @ w1.T + b1, 0).astype(np.float32)
            h2 = np.maximum(h1 @ w2.T + b2, 0).astype(np.float32)
            s1, s2 = scale(h1), scale(h2)
            layers = [_quantized_layer(w1, b1, s0, s1),
                      _quantized_layer(w2, b2, s1, s2),
                      _quantized_layer(w3, b3, s2)]
            common.update(activation_scales=[s0, s1, s2],
                          activation_calibration_max=[float(np.max(h1)), float(np.max(h2))],
                          layers=layers,
                          raw_parameter_bytes=64 + 4 + 12 + 12 + 12 + 160 + 80 + 8 + 2,
                          declared_local_scratch_bytes=28)
        result[family] = common
    for family in ("logistic", "mlp"):
        _, _, clip = infer(x_train, result[family], family)
        result[family]["train_clipping"] = clip
    return result


def _accumulate(activations, layer):
    weights = np.asarray(layer["weights"], dtype=np.int64)
    bias = np.asarray(layer["bias"], dtype=np.int64)
    acc = activations.astype(np.int64) @ weights.T + bias
    if np.max(np.abs(acc)) > np.iinfo(np.int32).max:
        raise ValueError("observed int32 accumulator overflow")
    return acc.astype(np.int32)


def infer(features, params, family):
    qx, input_clip, input_total = quantized_input(features, params)
    intermediate = {"input": qx}
    clipping = {"input": {"count": input_clip, "total": input_total,
                           "rate": input_clip / input_total}}
    if family == "logistic":
        logits = _accumulate(qx, params["layer"])
    else:
        first, second, final = params["layers"]
        acc1 = _accumulate(qx, first)
        act1, clip1, total1 = requantize_relu(acc1, first["multiplier"])
        acc2 = _accumulate(act1, second)
        act2, clip2, total2 = requantize_relu(acc2, second["multiplier"])
        logits = _accumulate(act2, final)
        intermediate.update(accumulator1=acc1, activation1=act1,
                            accumulator2=acc2, activation2=act2)
        clipping.update(activation1={"count": clip1, "total": total1, "rate": clip1 / total1},
                        activation2={"count": clip2, "total": total2, "rate": clip2 / total2})
    predictions = np.argmax(logits, axis=1).astype(np.uint8)
    intermediate.update(logits=logits, prediction=predictions)
    return predictions, intermediate, clipping


def c_float(value):
    text = f"{float(np.float32(value)):.9g}"
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def c_array(name, values, ctype):
    values = np.asarray(values)
    shape = "".join(f"[{n}]" for n in values.shape)
    if values.ndim == 2:
        lines = ["    {" + ", ".join(map(str, row)) + "}," for row in values]
    else:
        lines = ["    " + ", ".join(map(str, values)) + ","]
    return f"static const {ctype} {name}{shape} = {{\n" + "\n".join(lines) + "\n};\n"


def c_float_array(name, values):
    return c_array(name, [c_float(x) for x in values], "float")


def common_c(params):
    return (c_float_array("scaler_mean", params["scaler_mean"])
            + c_float_array("scaler_inv_std", params["scaler_inv_std"])
            + f"static const float input_inv_scale = {c_float(params['input_inv_scale'])};\n")


def render_logistic(params):
    layer = params["layer"]
    return ("/* Generated by training/quantize.py. Do not edit manually. */\n"
            '#include "model_api.h"\n#include "model_int8_debug.h"\n'
            '#include "quantized_ops.h"\n\n'
            + common_c(params)
            + f"const float teb_logistic_input_scale = {c_float(params['input_scale'])};\n"
            + f"const float teb_logistic_weight_scale = {c_float(layer['weight_scale'])};\n"
            + f"const float teb_logistic_bias_scale = {c_float(layer['bias_scale'])};\n"
            + c_array("weights", layer["weights"], "int8_t")
            + c_array("bias", layer["bias"], "int32_t")
            + """
static int infer(const float *features, teb_logistic_int8_debug_t *debug)
{
    int8_t input[8];
    for (int i = 0; i < 8; ++i) {
        float scaled = (features[i] - scaler_mean[i]) * scaler_inv_std[i];
        input[i] = teb_quantize_symmetric(scaled * input_inv_scale);
        if (debug) debug->input[i] = input[i];
    }
    int best = 0;
    int32_t best_score = 0;
    for (int c = 0; c < 4; ++c) {
        int32_t score = bias[c];
        for (int i = 0; i < 8; ++i)
            score += (int32_t)input[i] * weights[c][i];
        if (debug) debug->logits[c] = score;
        if (c == 0 || score > best_score) { best_score = score; best = c; }
    }
    if (debug) debug->prediction = best;
    return best;
}

int predict_logistic_int8(const float *features) { return infer(features, 0); }
#ifdef TEB_HOST_DEBUG
void teb_debug_logistic_int8(const float *features, teb_logistic_int8_debug_t *out)
{ infer(features, out); }
#endif
""")


def render_mlp(params):
    first, second, final = params["layers"]
    prefix = ("/* Generated by training/quantize.py. Do not edit manually. */\n"
              '#include "model_api.h"\n#include "model_int8_debug.h"\n'
              '#include "quantized_ops.h"\n\n' + common_c(params))
    for i, value in enumerate(params["activation_scales"]):
        prefix += f"const float teb_mlp_activation_scale{i} = {c_float(value)};\n"
    for i, layer in enumerate(params["layers"], 1):
        prefix += f"const float teb_mlp_weight_scale{i} = {c_float(layer['weight_scale'])};\n"
        prefix += f"const float teb_mlp_bias_scale{i} = {c_float(layer['bias_scale'])};\n"
        prefix += c_array(f"w{i}", layer["weights"], "int8_t")
        prefix += c_array(f"b{i}", layer["bias"], "int32_t")
    prefix += (f"static const int32_t multiplier1 = {first['multiplier']};\n"
               f"static const int32_t multiplier2 = {second['multiplier']};\n"
               f"static const uint8_t requant_shift1 = {SHIFT};\n"
               f"static const uint8_t requant_shift2 = {SHIFT};\n")
    return prefix + """
static int infer(const float *features, teb_mlp_int8_debug_t *debug)
{
    int8_t input[8], h1[8], h2[8];
    for (int i = 0; i < 8; ++i) {
        float scaled = (features[i] - scaler_mean[i]) * scaler_inv_std[i];
        input[i] = teb_quantize_symmetric(scaled * input_inv_scale);
        if (debug) debug->input[i] = input[i];
    }
    for (int j = 0; j < 8; ++j) {
        int32_t acc = b1[j];
        for (int i = 0; i < 8; ++i) acc += (int32_t)input[i] * w1[j][i];
        h1[j] = teb_requantize_relu(acc, multiplier1, requant_shift1);
        if (debug) { debug->accumulator1[j] = acc; debug->activation1[j] = h1[j]; }
    }
    for (int j = 0; j < 8; ++j) {
        int32_t acc = b2[j];
        for (int i = 0; i < 8; ++i) acc += (int32_t)h1[i] * w2[j][i];
        h2[j] = teb_requantize_relu(acc, multiplier2, requant_shift2);
        if (debug) { debug->accumulator2[j] = acc; debug->activation2[j] = h2[j]; }
    }
    int best = 0;
    int32_t best_score = 0;
    for (int c = 0; c < 4; ++c) {
        int32_t score = b3[c];
        for (int i = 0; i < 8; ++i) score += (int32_t)h2[i] * w3[c][i];
        if (debug) debug->logits[c] = score;
        if (c == 0 || score > best_score) { best_score = score; best = c; }
    }
    if (debug) debug->prediction = best;
    return best;
}

int predict_mlp_int8(const float *features) { return infer(features, 0); }
#ifdef TEB_HOST_DEBUG
void teb_debug_mlp_int8(const float *features, teb_mlp_int8_debug_t *out)
{ infer(features, out); }
#endif
"""


def render_expected(logistic_predictions, mlp_predictions):
    lines = ["/* Generated by training/quantize.py. Do not edit manually. */",
             '#include "benchmark_int8_expected.h"']
    for name, values in (("logistic", logistic_predictions), ("mlp", mlp_predictions)):
        lines.append(f"const uint8_t benchmark_expected_{name}_int8[BENCHMARK_SAMPLES] = {{")
        for start in range(0, len(values), 32):
            lines.append("    " + ", ".join(map(str, values[start:start + 32])) + ",")
        lines.append("};")
    return "\n".join(lines) + "\n"


def generate_artifacts():
    splits = train.load_data()
    x_train, _ = splits["train"]
    x_test, y_test = splits["test"]
    models = train.train_models(*splits["train"])
    params = calibrate(models, x_train)
    files = {
        MODELS / "model_logistic_int8.c": render_logistic(params["logistic"]),
        MODELS / "model_mlp_int8.c": render_mlp(params["mlp"]),
    }
    subset = selected_rows()[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    lp, _, _ = infer(subset, params["logistic"], "logistic")
    mp, _, _ = infer(subset, params["mlp"], "mlp")
    files[EXPECTED] = render_expected(lp.tolist(), mp.tolist())
    # All quantization constants and train calibration evidence are canonical.
    files[OUT / "quantization_params.json"] = json.dumps(params, indent=2, sort_keys=True) + "\n"
    rows, per_class, confusion = [], [], []
    for family, precision in (("rule", "native"), ("tree", "native"),
                              ("logistic", "fp32"), ("logistic", "int8"),
                              ("mlp", "fp32"), ("mlp", "int8")):
        if precision == "int8":
            predictions, _, clips = infer(x_test, params[family], family)
            raw_bytes = params[family]["raw_parameter_bytes"]
            scratch = params[family]["declared_local_scratch_bytes"]
        else:
            predictions = train.predict_ids(family, models[family], x_test)
            clips = {}
            size = train.estimate_size(family, models[family])
            raw_bytes = size["raw_constants_bytes"]
            scratch = None
        metrics = train.evaluate(y_test, predictions)
        variant = family if precision == "native" else f"{family}_{precision}"
        rows.append({"variant": variant, "model_family": family, "precision": precision,
                     "implementation": "portable_scalar", "quantization_scheme": params["scheme"] if precision == "int8" else "N/A",
                     "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                     "raw_parameter_bytes": raw_bytes if raw_bytes is not None else "N/A",
                     "declared_local_scratch_bytes": scratch if scratch is not None else "N/A",
                     **{f"{stage}_clip_rate": clips[stage]["rate"] if stage in clips else "N/A"
                        for stage in ("input", "activation1", "activation2")},
                     **{f"{stage}_clip_count": clips[stage]["count"] if stage in clips else "N/A"
                        for stage in ("input", "activation1", "activation2")}})
        for label, f1 in metrics["per_class_f1"].items():
            per_class.append({"variant": variant, "class": label, "f1": f1})
        for i, truth in enumerate(LABELS):
            for j, predicted in enumerate(LABELS):
                confusion.append({"variant": variant, "true": truth,
                                  "predicted": predicted, "count": int(metrics["confusion"][i, j])})
    base = {row["variant"]: row for row in rows}
    for row in rows:
        fp = base.get(f"{row['model_family']}_fp32")
        row["delta_accuracy_vs_fp32"] = row["accuracy"] - fp["accuracy"] if row["precision"] == "int8" else "N/A"
        row["delta_f1_vs_fp32"] = row["macro_f1"] - fp["macro_f1"] if row["precision"] == "int8" else "N/A"
    for name, table in (("offline_quantization.csv", rows),
                        ("offline_per_class_f1.csv", per_class),
                        ("offline_confusion.csv", confusion)):
        import io
        handle = io.StringIO(newline="")
        writer = csv.DictWriter(handle, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
        files[OUT / name] = handle.getvalue()
    return files


def main():
    for path, content in generate_artifacts().items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
