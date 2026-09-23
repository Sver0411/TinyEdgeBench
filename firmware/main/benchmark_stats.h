#ifndef TEB_BENCHMARK_STATS_H
#define TEB_BENCHMARK_STATS_H
#include <stddef.h>
#include <stdint.h>

/* Nearest-rank percentile on an ascending array. */
static inline uint32_t teb_percentile(const uint32_t *sorted, size_t count,
                                       unsigned numerator, unsigned denominator)
{
    size_t rank = (count * numerator + denominator - 1) / denominator;
    return sorted[rank - 1];
}
#endif
