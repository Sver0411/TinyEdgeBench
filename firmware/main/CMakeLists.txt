# Select exactly one predictor per hardware run. The baseline shares the
# harness and dataset but has no predictor.
#
#   idf.py -DTINYEDGEBENCH_MODEL=rule build

set(TINYEDGEBENCH_MODEL "rule" CACHE STRING "Predictor to link: baseline|rule|logistic|tree|mlp")

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
else()
    message(FATAL_ERROR "TINYEDGEBENCH_MODEL must be one of: baseline|rule|logistic|tree|mlp")
endif()

idf_component_register(
    SRCS
        "main.c"
        "benchmark.c"
        "benchmark_dataset.c"
        ${TINYEDGEBENCH_SRCS}
    INCLUDE_DIRS
        "."
        "models"
    REQUIRES
        "esp_timer"
        "esp_hw_support"
)

target_compile_definitions(${COMPONENT_TARGET} PRIVATE ${TINYEDGEBENCH_DEFS})

# The archive fallback keeps provenance when GitHub's Git endpoint is offline.
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
                OUTPUT_VARIABLE TEB_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT TEB_COMMIT)
    file(READ "${CMAKE_SOURCE_DIR}/../.upstream_commit" TEB_COMMIT)
    string(STRIP "${TEB_COMMIT}" TEB_COMMIT)
endif()
target_compile_definitions(${COMPONENT_TARGET} PRIVATE TEB_SOURCE_COMMIT="${TEB_COMMIT}")

file(GLOB_RECURSE TEB_SOURCE_FILES
    "${CMAKE_SOURCE_DIR}/main/*.c" "${CMAKE_SOURCE_DIR}/main/*.h"
    "${CMAKE_SOURCE_DIR}/../training/*.py")
list(APPEND TEB_SOURCE_FILES "${CMAKE_SOURCE_DIR}/../dataset/data.csv"
                             "${CMAKE_SOURCE_DIR}/../benchmark/export_dataset.py"
                             "${CMAKE_SOURCE_DIR}/../benchmark/measure_flash.py"
                             "${CMAKE_SOURCE_DIR}/../benchmark/parse_hardware_log.py"
                             "${CMAKE_SOURCE_DIR}/../benchmark/run_hardware.py"
                             "${CMAKE_SOURCE_DIR}/main/CMakeLists.txt")
list(FILTER TEB_SOURCE_FILES EXCLUDE REGEX "/\\._[^/]+$")
list(SORT TEB_SOURCE_FILES)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${TEB_SOURCE_FILES})
set(TEB_HASH_INPUT "")
foreach(TEB_FILE IN LISTS TEB_SOURCE_FILES)
    file(SHA256 "${TEB_FILE}" TEB_FILE_HASH)
    string(APPEND TEB_HASH_INPUT "${TEB_FILE_HASH}")
endforeach()
file(SHA256 "${CMAKE_SOURCE_DIR}/sdkconfig" TEB_CONFIG_HASH)
string(APPEND TEB_HASH_INPUT "${TEB_CONFIG_HASH}")
string(SHA256 TEB_SOURCE_HASH "${TEB_HASH_INPUT}")
if(TINYEDGEBENCH_SRCS)
    file(SHA256 "${CMAKE_SOURCE_DIR}/main/${TINYEDGEBENCH_SRCS}" TEB_MODEL_HASH)
else()
    set(TEB_MODEL_HASH "none")
endif()
target_compile_definitions(${COMPONENT_TARGET} PRIVATE
    TEB_SOURCE_HASH="${TEB_SOURCE_HASH}" TEB_MODEL_HASH="${TEB_MODEL_HASH}"
    TEB_CONFIG_HASH="${TEB_CONFIG_HASH}")
