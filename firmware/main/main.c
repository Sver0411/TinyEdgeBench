/* TinyEdgeBench ESP32 demo.
 *
 * Runs the four exported inference methods on a few fixed feature rows and
 * prints the results. No sensors, no Wi-Fi, no tasks - just inference.
 *
 * The latency hook at the bottom is a placeholder for when a board is
 * available: nothing has been measured on hardware yet.
 */

#include <stdio.h>

#include "esp_timer.h"
#include "models/model_api.h"

/* Four rows taken from the test split of dataset/data.csv (one per class). */
static const float demo_features[][NUM_FEATURES] = {
    {23.372562f, 51.017747f, 434.932639f, -0.098815f, -0.526816f, 1.950311f, 0.968099f, 0.121226f},   /* NORMAL */
    {22.121322f, 39.824396f, 213.925310f, -0.138429f, -1.243529f, -35.420706f, 1.579823f, 0.680169f}, /* RAPID_CHANGE */
    {24.972027f, 44.444505f, 444.575250f, -0.141309f, -1.594383f, -15.322257f, 1.682481f, 0.640560f}, /* SLOW_DRIFT */
    {21.647165f, 45.921643f, 376.171198f, -0.245404f, 0.021942f, 34.874015f, 2.458802f, 0.513554f},   /* NOISY */
};

typedef int (*predict_fn)(const float *);

static const struct {
    const char *name;
    predict_fn predict;
} methods[] = {
    {"rule", predict_rule},
    {"logistic", predict_logistic},
    {"tree", predict_tree},
    {"mlp", predict_mlp},
};

static void run_demo(void)
{
    for (size_t row = 0; row < sizeof(demo_features) / sizeof(demo_features[0]); row++) {
        printf("row %u:", (unsigned)row + 1);
        for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
            int cls = methods[m].predict(demo_features[row]);
            printf("  %s=%s", methods[m].name, class_names[cls]);
        }
        printf("\n");
    }
}

/* Reserved for on-board timing. Reported values are whatever the board
   measures; no hardware run has happened yet, so nothing is published. */
static void measure_latency(const float *features)
{
    const int iterations = 1000;
    volatile int sink = 0;

    for (size_t m = 0; m < sizeof(methods) / sizeof(methods[0]); m++) {
        uint64_t start = esp_timer_get_time();
        for (int i = 0; i < iterations; i++) {
            sink += methods[m].predict(features);
        }
        uint64_t elapsed = esp_timer_get_time() - start;

        printf("%-9s %llu.%03llu us/call (%d iterations)\n",
               methods[m].name,
               (unsigned long long)(elapsed / iterations),
               (unsigned long long)((elapsed % iterations) * 1000 / iterations),
               iterations);
    }
    (void)sink;
}

void app_main(void)
{
    printf("TinyEdgeBench firmware demo\n");
    run_demo();
    printf("\nlatency (real hardware only, nothing measured yet):\n");
    measure_latency(demo_features[0]);
}
