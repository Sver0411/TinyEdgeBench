/* One measurement path for every model. No serial I/O or allocation in the loop. */
#include "benchmark.h"
#include "benchmark_stats.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_idf_version.h"
#include "esp_timer.h"
#include "esp_private/esp_clk.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#ifndef TEB_GIT_COMMIT
#define TEB_GIT_COMMIT "unknown"
#endif
#ifndef TEB_GIT_DIRTY
#define TEB_GIT_DIRTY true
#endif
#ifndef TEB_SOURCE_HASH
#define TEB_SOURCE_HASH "unknown"
#endif
#ifndef TEB_MODEL_HASH
#define TEB_MODEL_HASH "unknown"
#endif
#ifndef TEB_CONFIG_HASH
#define TEB_CONFIG_HASH "unknown"
#endif

#define WARMUP 100
#define INFERENCES_PER_BATCH 64
#define TIMED_BATCHES 2000
#define ITERATIONS (INFERENCES_PER_BATCH * TIMED_BATCHES)

/* One timer interval covers 64 distinct input rows. */
static uint32_t batch_durations_us[TIMED_BATCHES];
static volatile uint32_t prediction_sink;

static int compare_u32(const void *left, const void *right)
{
    uint32_t a = *(const uint32_t *)left;
    uint32_t b = *(const uint32_t *)right;
    return (a > b) - (a < b);
}

static double percentile_us(unsigned numerator, unsigned denominator)
{
    return (double)teb_percentile(batch_durations_us, TIMED_BATCHES,
                                  numerator, denominator) / INFERENCES_PER_BATCH;
}

void run_benchmark(const char *model, predict_fn predict,
                   const uint8_t expected[BENCHMARK_SAMPLES])
{
    esp_chip_info_t chip;
    esp_chip_info(&chip);
#if defined(CONFIG_COMPILER_OPTIMIZATION_PERF)
    const char *optimization = "performance";
#elif defined(CONFIG_COMPILER_OPTIMIZATION_SIZE)
    const char *optimization = "size";
#elif defined(CONFIG_COMPILER_OPTIMIZATION_NONE)
    const char *optimization = "none";
#elif defined(CONFIG_COMPILER_OPTIMIZATION_DEBUG)
    const char *optimization = "debug -Og";
#else
    const char *optimization = "other";
#endif
#if defined(CONFIG_PM_ENABLE)
    const char *power_management = "enabled";
#else
    const char *power_management = "disabled";
#endif
    printf("TEB_BEGIN {\"model\":\"%s\",\"board\":\"unspecified\","
           "\"chip\":\"ESP32-S3\",\"revision\":%u,"
           "\"idf\":\"%s\",\"compiler\":\"GCC %s\","
           "\"cpu_mhz\":%u,\"optimization\":\"%s\","
           "\"image_flash_size\":\"%s\","
           "\"git_commit\":\"%s\",\"git_dirty\":%s,"
           "\"source_sha256\":\"%s\","
           "\"model_sha256\":\"%s\",\"sdkconfig_sha256\":\"%s\","
           "\"dataset_sha256\":\"%s\",\"subset_sha256\":\"%s\","
           "\"samples\":%u,\"warmup\":%u,\"iterations\":%u,"
           "\"timed_batches\":%u,\"inferences_per_batch\":%u,"
           "\"wifi\":\"disabled\",\"bluetooth\":\"disabled\","
           "\"power_management\":\"%s\",\"log_level\":%u}\n",
           model, chip.revision, esp_get_idf_version(), __VERSION__,
           (unsigned)(esp_clk_cpu_freq() / 1000000), optimization,
           CONFIG_ESPTOOLPY_FLASHSIZE,
           TEB_GIT_COMMIT, TEB_GIT_DIRTY ? "true" : "false",
           TEB_SOURCE_HASH, TEB_MODEL_HASH,
           TEB_CONFIG_HASH, benchmark_dataset_sha256, benchmark_subset_sha256,
           BENCHMARK_SAMPLES, WARMUP, ITERATIONS,
           TIMED_BATCHES, INFERENCES_PER_BATCH, power_management,
           (unsigned)CONFIG_LOG_DEFAULT_LEVEL);

    uint32_t confusion[NUM_CLASSES][NUM_CLASSES] = {{0}};
    unsigned correct = 0;
    for (unsigned i = 0; i < BENCHMARK_SAMPLES; i++) {
        int prediction = predict(benchmark_features[i]);
        if (prediction < 0 || prediction >= NUM_CLASSES || prediction != expected[i]) {
            printf("TEB_ERROR {\"reason\":\"prediction_mismatch\","
                   "\"sample\":%u,\"expected\":%u,\"actual\":%d}\n",
                   i, expected[i], prediction);
            return;
        }
        int label = benchmark_labels[i];
        confusion[label][prediction]++;
        correct += label == prediction;
    }

    for (unsigned i = 0; i < WARMUP; i++) {
        prediction_sink += (unsigned)predict(benchmark_features[i % BENCHMARK_SAMPLES]);
    }

    size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    int64_t total_batch_us = 0;
    uint32_t checksum = 0;
    for (unsigned batch = 0; batch < TIMED_BATCHES; batch++) {
        int64_t start = esp_timer_get_time();
        for (unsigned j = 0; j < INFERENCES_PER_BATCH; j++) {
            const float *features = benchmark_features[
                (batch * INFERENCES_PER_BATCH + j) % BENCHMARK_SAMPLES];
            checksum += (uint32_t)predict(features);
        }
        int64_t end = esp_timer_get_time();
        uint32_t elapsed = (uint32_t)(end - start);
        batch_durations_us[batch] = elapsed;
        total_batch_us += elapsed;
    }
    prediction_sink += checksum;
    size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    size_t heap_minimum = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT);
    /* ESP-IDF's xTaskGetStackHighWaterMark reports bytes, not FreeRTOS words. */
    unsigned stack_remaining = (unsigned)uxTaskGetStackHighWaterMark(NULL);
    qsort(batch_durations_us, TIMED_BATCHES,
          sizeof(batch_durations_us[0]), compare_u32);

    printf("TEB_RESULT {\"model\":\"%s\",\"correct\":%u,\"samples\":%u,"
           "\"confusion\":[", model, correct, BENCHMARK_SAMPLES);
    for (unsigned truth = 0; truth < NUM_CLASSES; truth++) {
        for (unsigned pred = 0; pred < NUM_CLASSES; pred++) {
            printf("%s%" PRIu32, (truth || pred) ? "," : "", confusion[truth][pred]);
        }
    }
    printf("],\"checksum\":%" PRIu32 ",\"latency_min_us\":%.6f"
           ",\"latency_mean_us\":%.6f,\"latency_p50_us\":%.6f"
           ",\"latency_p95_us\":%.6f,\"latency_p99_us\":%.6f"
           ",\"latency_max_us\":%.6f,\"heap_before_bytes\":%u,"
           "\"heap_after_bytes\":%u,\"heap_minimum_bytes\":%u,"
           "\"main_task_stack_configured_bytes\":%u,"
           "\"main_task_stack_minimum_remaining_bytes\":%u}\n",
           checksum, (double)batch_durations_us[0] / INFERENCES_PER_BATCH,
           (double)total_batch_us / ITERATIONS,
           percentile_us(50, 100), percentile_us(95, 100),
           percentile_us(99, 100),
           (double)batch_durations_us[TIMED_BATCHES - 1] / INFERENCES_PER_BATCH,
           (unsigned)heap_before,
           (unsigned)heap_after, (unsigned)heap_minimum,
           (unsigned)CONFIG_ESP_MAIN_TASK_STACK_SIZE, stack_remaining);
    printf("TEB_END {\"model\":\"%s\"}\n", model);
}
