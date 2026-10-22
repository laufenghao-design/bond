#!/bin/bash

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

cd "$PROJECT_ROOT"

GQADIR="./playground/data/eval/gqa/data"
cd $GQADIR
python eval.py --tier testdev_balanced_64  --questions testdev_balanced_questions.json

cd "$PROJECT_ROOT"

cd ./playground/data/eval/MME
cd eval_tool
python calculation.py --results_dir answers/llava-v1.5-7b_64

cd "$PROJECT_ROOT"

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b_64.jsonl


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_64.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_64_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b_64_result.json


python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b_64.jsonl

