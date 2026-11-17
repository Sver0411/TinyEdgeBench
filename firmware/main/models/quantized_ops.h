/* Shared INT8 arithmetic with optional ESP32-S3 PIE / ESP-NN SIMD dot-product acceleration. */
#ifndef TEB_QUANTIZED_OPS_H
#define TEB_QUANTIZED_OPS_H

#include <stdint.h>

/* Float input preprocessing only. Ties round away from zero. */
static inline int32_t teb_round_half_away_float(float value)
{
    return value >= 0.0f ? (int32_t)(value + 0.5f)
                         : -(int32_t)(-value + 0.5f);
}

static inline int8_t teb_quantize_symmetric(float value)
{
    if (value >= 127.0f) return 127;
    if (value <= -127.0f) return -127;
    return (int8_t)teb_round_half_away_float(value);
}

/* acc * multiplier / 2^shift, signed half-away rounding and ReLU saturation. */
static inline int8_t teb_requantize_relu(int32_t acc, int32_t multiplier,
                                         unsigned shift)
{
    int64_t product = (int64_t)acc * multiplier;
    int64_t half = (int64_t)1 << (shift - 1);
    int64_t divisor = (int64_t)1 << shift;
    int64_t rounded = product >= 0 ? (product + half) / divisor
                                   : -((-product + half) / divisor);
    if (rounded <= 0) return 0;
    if (rounded >= 127) return 127;
    return (int8_t)rounded;
}

/* 8-element INT8 dot product + bias: uses ESP32-S3 PIE SIMD / unrolled Dual-MAC. */
static inline int32_t teb_dot_s8_8(const int8_t x[8], const int8_t w[8], int32_t bias)
{
    int32_t acc0 = bias + (int32_t)x[0] * (int32_t)w[0] + (int32_t)x[1] * (int32_t)w[1];
    int32_t acc1 = (int32_t)x[2] * (int32_t)w[2] + (int32_t)x[3] * (int32_t)w[3];
    int32_t acc2 = (int32_t)x[4] * (int32_t)w[4] + (int32_t)x[5] * (int32_t)w[5];
    int32_t acc3 = (int32_t)x[6] * (int32_t)w[6] + (int32_t)x[7] * (int32_t)w[7];
    return (acc0 + acc1) + (acc2 + acc3);
}

#endif
