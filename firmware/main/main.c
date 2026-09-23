#include <stdio.h>
#include "benchmark.h"
#include "models/model_api.h"

const char *const class_names[NUM_CLASSES] = {
    "NORMAL", "RAPID_CHANGE", "SLOW_DRIFT", "NOISY"
};

#if !defined(TINYEDGEBENCH_HAVE_RULE) && !defined(TINYEDGEBENCH_HAVE_LOGISTIC) && \
    !defined(TINYEDGEBENCH_HAVE_TREE) && !defined(TINYEDGEBENCH_HAVE_MLP)
static int predict_baseline(const float *features)
{
    (void)features;
    return CLASS_NORMAL;
}
#endif

void app_main(void)
{
#if defined(TINYEDGEBENCH_HAVE_RULE)
    run_benchmark("rule", predict_rule, benchmark_expected_rule);
#elif defined(TINYEDGEBENCH_HAVE_LOGISTIC)
    run_benchmark("logistic", predict_logistic, benchmark_expected_logistic);
#elif defined(TINYEDGEBENCH_HAVE_TREE)
    run_benchmark("tree", predict_tree, benchmark_expected_tree);
#elif defined(TINYEDGEBENCH_HAVE_MLP)
    run_benchmark("mlp", predict_mlp, benchmark_expected_mlp);
#else
    printf("TEB_BASELINE no model linked; sizing reference only\n");
    run_benchmark("baseline", predict_baseline, benchmark_expected_rule);
#endif
}
