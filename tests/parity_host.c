/* Host-side harness for the Python/C parity test.
 *
 * Reads one feature row per line and prints the four C predictions:
 *     rule logistic tree mlp
 *
 * Usage: parity_host samples.txt
 */

#include <stdio.h>

#include "models/model_api.h"

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s samples.txt\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (!file) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }

    float features[NUM_FEATURES];
    while (1) {
        for (int i = 0; i < NUM_FEATURES; i++) {
            if (fscanf(file, "%f", &features[i]) != 1) {
                fclose(file);
                return 0;
            }
        }
        printf("%d %d %d %d\n",
               predict_rule(features),
               predict_logistic(features),
               predict_tree(features),
               predict_mlp(features));
    }
}
