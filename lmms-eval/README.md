# LMMS-Eval image evaluation subset

This directory contains the local evaluation framework used by BOND's image
evaluation entrypoints. The model registry exposes `llava` and `qwen2_5_vl`.
The bundled image tasks retain their original prompts, splits, result parsers,
and metric implementations.

Install this directory in the Qwen environment with:

```bash
pip install -e ./lmms-eval
```

Run that command from the BOND repository root. See the main
[README](../README.md) for environment versions, reference generation, dataset
layout, and evaluation commands. Shared framework utilities keep their
original implementations and imports.

This subset derives from the LMMs-Lab evaluation framework. Upstream project
information is available in the original repository:
https://github.com/EvolvingLMMs-Lab/lmms-eval.
Original source notices and the [MIT license](LICENSE) are retained.
