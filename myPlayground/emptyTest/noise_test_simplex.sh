#!/bin/bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

cd "$PROJECT_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export REFERENCE_SCORE_FILE="${REFERENCE_SCORE_FILE:-${PROJECT_ROOT}/myPlayground/emptyTest/output/description/relation_vis_text_empty_with_description_Simplex_64.jsonl}"
export NOISE_SCALE=64

python myPlayground/emptyTest/noise_test_simplex.py \
    --model-path "${LLAVA_MODEL_PATH:-liuhaotian/llava-v1.5-7b}" \
    --num-images 1000 \
    --prompt-text "Describe this image." \
    --answers-file ./myPlayground/emptyTest/output/noise_simplex_test_results.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --retained_tokens 64
