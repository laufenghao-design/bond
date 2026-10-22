# Contributing

This repository covers offline reference-score generation, LLaVA-1.5 image
evaluation, and Qwen2.5-VL image evaluation. Keep the existing directory layout
and entrypoints when making changes.

Use Python 3.10 or later for development. Formatting and syntax checks can run
in a lightweight environment containing only Ruff:

```bash
python -m pip install ruff==0.14.3
ruff format --no-cache --check .
ruff check --no-cache .
rg --files --hidden -g '*.sh' -0 | xargs -0 -r -n 1 bash -n
```

Apply formatting with `ruff format --no-cache .`. The root `pyproject.toml`
defines the formatter settings and the syntax checks. These checks do not load
model weights or require a GPU.

Run the alignment tests in either model environment with PyTorch installed:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
```

These CPU tests check attention statistics, original-grid reference lookup,
greedy selection, full-token reference recording, and the dense Qwen setting.
They load individual functions and use small tensors without model checkpoints.

Keep formatting and documentation changes separate from changes to score
computation, normalization, pruning, generation, prompts, or evaluation metrics.
For changes to model behavior, describe the affected settings and the validation
performed. Preserve upstream copyright notices and licenses.

Supply checkpoints, datasets, generated probes, reference banks, predictions,
and local environment settings outside version control. Document any new
configuration in the main README.
