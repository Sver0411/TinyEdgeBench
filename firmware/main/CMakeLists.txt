# Select which predictor(s) get linked into the firmware.
#
# Default "all" builds the normal demo. benchmark/measure_flash.py builds
# "baseline" plus one variant per method, to measure how much compiled flash
# each predictor actually adds.
#
#   idf.py -DTINYEDGEBENCH_MODEL=rule build

set(TINYEDGEBENCH_MODEL "all" CACHE STRING "Predictor(s) to link: all|baseline|rule|logistic|tree|mlp")

if(TINYEDGEBENCH_MODEL STREQUAL "baseline")
    set(TINYEDGEBENCH_SRCS "")
    set(TINYEDGEBENCH_DEFS "")
elseif(TINYEDGEBENCH_MODEL STREQUAL "rule")
    set(TINYEDGEBENCH_SRCS "models/model_rule.c")
    set(TINYEDGEBENCH_DEFS TINYEDGEBENCH_HAVE_RULE)
elseif(TINYEDGEBENCH_MODEL STREQUAL "logistic")
    set(TINYEDGEBENCH_SRCS "models/model_logistic.c")
    set(TINYEDGEBENCH_DEFS TINYEDGEBENCH_HAVE_LOGISTIC)
elseif(TINYEDGEBENCH_MODEL STREQUAL "tree")
    set(TINYEDGEBENCH_SRCS "models/model_tree.c")
    set(TINYEDGEBENCH_DEFS TINYEDGEBENCH_HAVE_TREE)
elseif(TINYEDGEBENCH_MODEL STREQUAL "mlp")
    set(TINYEDGEBENCH_SRCS "models/model_mlp.c")
    set(TINYEDGEBENCH_DEFS TINYEDGEBENCH_HAVE_MLP)
elseif(TINYEDGEBENCH_MODEL STREQUAL "all")
    set(TINYEDGEBENCH_SRCS
        "models/model_rule.c"
        "models/model_logistic.c"
        "models/model_tree.c"
        "models/model_mlp.c")
    set(TINYEDGEBENCH_DEFS
        TINYEDGEBENCH_HAVE_RULE
        TINYEDGEBENCH_HAVE_LOGISTIC
        TINYEDGEBENCH_HAVE_TREE
        TINYEDGEBENCH_HAVE_MLP)
else()
    message(FATAL_ERROR "TINYEDGEBENCH_MODEL must be one of: all|baseline|rule|logistic|tree|mlp")
endif()

idf_component_register(
    SRCS
        "main.c"
        ${TINYEDGEBENCH_SRCS}
    INCLUDE_DIRS
        "."
        "models"
    REQUIRES
        "esp_timer"
)

target_compile_definitions(${COMPONENT_TARGET} PRIVATE ${TINYEDGEBENCH_DEFS})
