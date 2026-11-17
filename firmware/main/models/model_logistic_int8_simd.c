/* ESP-NN / PIE SIMD-accelerated INT8 multinomial logistic regression. */
#include "model_api.h"
#include "quantized_ops.h"

static const float scaler_mean[8] = {
    23.8940926f, 49.5312157f, 298.455536f, -0.00198788778f, -0.0193460956f, -0.068763487f, 2.12744904f, 0.497095823f,
};
static const float scaler_inv_std[8] = {
    0.452007145f, 0.109764814f, 0.0100932177f, 2.36836481f, 0.451575547f, 0.047104314f, 0.61937505f, 2.35100341f,
};
static const float input_inv_scale = 13.6932974f;
static const int8_t weights[4][8] = {
    {-1, -1, 0, 0, 0, 0, -51, -65},
    {0, 0, 0, 0, 0, 0, -3, 41},
    {1, -1, 0, -1, 1, 0, -73, 28},
    {0, 1, 0, 1, -2, 0, 127, -4},
};
static const int32_t bias[4] = {
    -739, 330, -99, 508,
};

int predict_logistic_int8_simd(const float *features)
{
    int8_t input[8] __attribute__((aligned(16)));
    for (int i = 0; i < 8; ++i) {
        float scaled = (features[i] - scaler_mean[i]) * scaler_inv_std[i];
        input[i] = teb_quantize_symmetric(scaled * input_inv_scale);
    }
    int32_t s0 = teb_dot_s8_8(input, weights[0], bias[0]);
    int32_t s1 = teb_dot_s8_8(input, weights[1], bias[1]);
    int32_t s2 = teb_dot_s8_8(input, weights[2], bias[2]);
    int32_t s3 = teb_dot_s8_8(input, weights[3], bias[3]);
    int best = 0;
    int32_t best_score = s0;
    if (s1 > best_score) { best_score = s1; best = 1; }
    if (s2 > best_score) { best_score = s2; best = 2; }
    if (s3 > best_score) { best_score = s3; best = 3; }
    return best;
}
