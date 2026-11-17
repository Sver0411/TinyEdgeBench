/* ESP-NN / PIE SIMD-accelerated INT8 8->8->8->4 MLP classifier. */
#include "model_api.h"
#include "quantized_ops.h"

static const float scaler_mean[8] = {
    23.8940926f, 49.5312157f, 298.455536f, -0.00198788778f, -0.0193460956f, -0.068763487f, 2.12744904f, 0.497095823f,
};
static const float scaler_inv_std[8] = {
    0.452007145f, 0.109764814f, 0.0100932177f, 2.36836481f, 0.451575547f, 0.047104314f, 0.61937505f, 2.35100341f,
};
static const float input_inv_scale = 13.6932974f;
static const int8_t w1[8][8] = {
    {-33, -16, -26, 15, -4, -12, 16, 61},
    {-6, -12, -5, -3, 2, 10, -127, -4},
    {-9, -4, -7, -14, 32, -35, 108, -19},
    {-3, 7, 2, -6, -2, 9, 7, -94},
    {-3, -3, -2, 18, -26, 2, 84, -27},
    {3, 2, 4, -12, 2, 27, 71, 43},
    {0, 9, -1, 2, -8, -10, -45, 59},
    {0, -47, 21, -8, 41, -19, 8, 32},
};
static const int32_t b1[8] = {
    -51, 463, 660, 560, 597, 1332, 643, -586,
};
static const int8_t w2[8][8] = {
    {-27, -47, 104, 23, 127, 70, -46, -57},
    {10, 41, -35, -65, -74, 17, 50, -34},
    {14, -10, -35, -43, -15, 6, -23, -13},
    {62, -107, 50, -29, 65, 98, 1, 24},
    {70, -41, 31, -57, -1, 100, 63, -4},
    {0, -16, 2, 10, -1, -2, -7, -11},
    {6, 42, -15, -58, -27, 38, 91, 56},
    {-59, 74, 6, 31, -54, -17, 44, -12},
};
static const int32_t b2[8] = {
    -217, 257, -208, 680, 987, -296, 282, -59,
};
static const int8_t w3[4][8] = {
    {-34, -76, -34, -47, -87, -16, -127, 60},
    {-101, -26, -10, 53, 68, 10, 32, -26},
    {-22, 76, 47, -84, 12, -12, 43, -18},
    {61, -56, -21, 80, 44, -24, -64, -89},
};
static const int32_t b3[4] = {
    532, -181, 133, -345,
};
static const int32_t multiplier1 = 177364;
static const int32_t multiplier2 = 93324;
static const uint8_t requant_shift1 = 24;
static const uint8_t requant_shift2 = 24;

int predict_mlp_int8_simd(const float *features)
{
    int8_t input[8] __attribute__((aligned(16)));
    int8_t h1[8] __attribute__((aligned(16)));
    int8_t h2[8] __attribute__((aligned(16)));
    for (int i = 0; i < 8; ++i) {
        float scaled = (features[i] - scaler_mean[i]) * scaler_inv_std[i];
        input[i] = teb_quantize_symmetric(scaled * input_inv_scale);
    }
    for (int j = 0; j < 8; ++j) {
        int32_t acc = teb_dot_s8_8(input, w1[j], b1[j]);
        h1[j] = teb_requantize_relu(acc, multiplier1, requant_shift1);
    }
    for (int j = 0; j < 8; ++j) {
        int32_t acc = teb_dot_s8_8(h1, w2[j], b2[j]);
        h2[j] = teb_requantize_relu(acc, multiplier2, requant_shift2);
    }
    int best = 0;
    int32_t best_score = teb_dot_s8_8(h2, w3[0], b3[0]);
    for (int c = 1; c < 4; ++c) {
        int32_t score = teb_dot_s8_8(h2, w3[c], b3[c]);
        if (score > best_score) { best_score = score; best = c; }
    }
    return best;
}
