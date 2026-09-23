#ifndef BENCHMARK_DATASET_H
#define BENCHMARK_DATASET_H

#include <stdint.h>
#include "models/model_api.h"

#define BENCHMARK_SAMPLES 256
extern const char benchmark_dataset_sha256[];
extern const char benchmark_subset_sha256[];
extern const float benchmark_features[BENCHMARK_SAMPLES][NUM_FEATURES];
extern const uint8_t benchmark_labels[BENCHMARK_SAMPLES];
extern const uint8_t benchmark_expected_rule[BENCHMARK_SAMPLES];
extern const uint8_t benchmark_expected_logistic[BENCHMARK_SAMPLES];
extern const uint8_t benchmark_expected_tree[BENCHMARK_SAMPLES];
extern const uint8_t benchmark_expected_mlp[BENCHMARK_SAMPLES];

#endif
