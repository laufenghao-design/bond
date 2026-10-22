#!/bin/bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"
cd "$PROJECT_ROOT" || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Checkpoint configuration.
model_id="${QWEN_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
model_name="Qwen2.5-VL-7B-Instruct"


output_path="./logs/${model_name}/${task}/"
mkdir -p "$output_path"

Sparse=${1:-True}
export RETENTION_RATIO="${2:-${RETENTION_RATIO:-0.1}}"

# Run the image benchmarks with the current model adapter.
# ai2d,chartqa,docvqa_val,pope,textvqa_val,mme,ocrbench,scienceqa,mmmu_val
python3 -m accelerate.commands.launch \
    --num_processes="${NUM_PROCESSES:-1}" \
    --main_process_port 50008 \
    -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=$model_id,device_map=cuda,use_flash_attention_2=True,Sparse=$Sparse \
    --tasks  ai2d,chartqa,docvqa_val,pope,textvqa_val,mme,ocrbench,scienceqa,mmmu_val\
    --batch_size 1 \
    --log_samples \
    --output_path "$output_path"
