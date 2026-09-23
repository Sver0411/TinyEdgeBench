/* Shared, portable scalar INT8 arithmetic. No implementation-specific SIMD. */
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

#endif
