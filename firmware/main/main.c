#include <stdio.h>
#include "benchmark.h"
#include "benchmark_int8_expected.h"
#include "models/model_api.h"

const char *const class_names[NUM_CLASSES] = {
    "NORMAL", "RAPID_CHANGE", "SLOW_DRIFT", "NOISY"
};

#if !defined(TINYEDGEBENCH_HAVE_RULE) && !defined(TINYEDGEBENCH_HAVE_LOGISTIC) && \
    !defined(TINYEDGEBENCH_HAVE_TREE) && !defined(TINYEDGEBENCH_HAVE_MLP) && \
    !defined(TINYEDGEBENCH_HAVE_LOGISTIC_INT8) && !defined(TINYEDGEBENCH_HAVE_MLP_INT8) && \
    !defined(TINYEDGEBENCH_HAVE_LOGISTIC_INT8_SIMD) && !defined(TINYEDGEBENCH_HAVE_MLP_INT8_SIMD)
static int predict_timing_baseline(const float *features)
{
    return features[0] > 1e30f ? 1 : 0;
}
#endif

void app_main(void)
{
#if defined(TINYEDGEBENCH_HAVE_RULE)
    run_benchmark("rule", predict_rule, benchmark_expected_rule);
#elif defined(TINYEDGEBENCH_HAVE_LOGISTIC)
    run_benchmark("logistic_fp32", predict_logistic, benchmark_expected_logistic);
#elif defined(TINYEDGEBENCH_HAVE_LOGISTIC_INT8)
    run_benchmark("logistic_int8", predict_logistic_int8, benchmark_expected_logistic_int8);
#elif defined(TINYEDGEBENCH_HAVE_LOGISTIC_INT8_SIMD)
    run_benchmark("logistic_int8_simd", predict_logistic_int8_simd, benchmark_expected_logistic_int8);
#elif defined(TINYEDGEBENCH_HAVE_TREE)
    run_benchmark("tree", predict_tree, benchmark_expected_tree);
#elif defined(TINYEDGEBENCH_HAVE_MLP)
    run_benchmark("mlp_fp32", predict_mlp, benchmark_expected_mlp);
#elif defined(TINYEDGEBENCH_HAVE_MLP_INT8)
    run_benchmark("mlp_int8", predict_mlp_int8, benchmark_expected_mlp_int8);
#elif defined(TINYEDGEBENCH_HAVE_MLP_INT8_SIMD)
    run_benchmark("mlp_int8_simd", predict_mlp_int8_simd, benchmark_expected_mlp_int8);
#else
    run_benchmark("timing_baseline", predict_timing_baseline, NULL);
#endif
}
