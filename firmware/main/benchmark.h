#ifndef TINYEDGEBENCH_BENCHMARK_H
#define TINYEDGEBENCH_BENCHMARK_H

#include <stdint.h>
#include "benchmark_dataset.h"

typedef int (*predict_fn)(const float *);

void run_benchmark(const char *model, predict_fn predict,
                   const uint8_t expected[BENCHMARK_SAMPLES]);

#endif
