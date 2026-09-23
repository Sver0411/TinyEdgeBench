/* Host-only intermediate-state API; excluded from production builds. */
#ifndef TEB_MODEL_INT8_DEBUG_H
#define TEB_MODEL_INT8_DEBUG_H
#include <stdint.h>

typedef struct {
    int8_t input[8];
    int32_t logits[4];
    int prediction;
} teb_logistic_int8_debug_t;

typedef struct {
    int8_t input[8];
    int32_t accumulator1[8];
    int8_t activation1[8];
    int32_t accumulator2[8];
    int8_t activation2[8];
    int32_t logits[4];
    int prediction;
} teb_mlp_int8_debug_t;

#ifdef TEB_HOST_DEBUG
void teb_debug_logistic_int8(const float *features, teb_logistic_int8_debug_t *out);
void teb_debug_mlp_int8(const float *features, teb_mlp_int8_debug_t *out);
#endif
#endif
