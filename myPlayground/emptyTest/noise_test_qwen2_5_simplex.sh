#!/bin/bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Set MODE: "generate" to create images, "eval" to evaluate
MODE="${MODE:-eval}"  # Change to "eval" to run evaluation

# Image settings
IMAGE_DIR="${PROJECT_ROOT}/myPlayground/emptyTest/simplex_image_Qwen"
NUM_IMAGES=${NUM_IMAGES:-1000}
export NOISE_SCALE=64
export REFERENCE_SCORE_FILE="${REFERENCE_SCORE_FILE:-${PROJECT_ROOT}/myPlayground/emptyTest/output/description/relation_vis_text_empty_with_description_Simplex_64_qwen.jsonl}"
IMAGE_SIZE=640

if [ "$MODE" = "generate" ]; then
    echo "Generating Simplex noise images with scale $NOISE_SCALE..."
    python3 myPlayground/emptyTest/noise_test_qwen2_5_simplex.py \
        --mode generate \
        --image-dir $IMAGE_DIR \
        --num-images $NUM_IMAGES \
        --image-size $IMAGE_SIZE

elif [ "$MODE" = "eval" ]; then
    echo "Evaluating model on Simplex noise images with scale $NOISE_SCALE..."
    python3 myPlayground/emptyTest/noise_test_qwen2_5_simplex.py \
        --mode eval \
        --image-dir $IMAGE_DIR \
        --model-path "${QWEN_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" \
        --prompt-text "Describe this image." \
        --answers-file ./myPlayground/emptyTest/output/noise_qwen2_5_simplex_test_results_scale_${NOISE_SCALE}.jsonl \
        --temperature 0 \
        --max_new_tokens 128
else
    echo "Invalid MODE: $MODE. Use 'generate' or 'eval'"
    exit 1
fi
