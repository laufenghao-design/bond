"""Raw, full-key-normalized reference statistics for LLaVA image prefill."""

import torch


def reference_scores(
    attention_weights: torch.Tensor,
    v_token_start: int,
    v_token_num: int,
    text_token_start: int,
    t_token_idx: torch.Tensor,
) -> torch.Tensor:
    """Average text-to-visual attention over all heads and instruction queries.

    The caller supplies probabilities normalized over all causally visible
    keys. Reference recording retains every visual token at every layer.
    """
    attention_mean = attention_weights.mean(1)
    text_rows = t_token_idx[1] + text_token_start
    return attention_mean[:, text_rows, v_token_start : v_token_start + v_token_num].mean(1)
