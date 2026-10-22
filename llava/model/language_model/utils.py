import torch
import math
import os
import csv
from pathlib import Path

EPSILON = 1e-6
_MEAN_SCORE_CACHE = {}
_EXPECTED_DEBIAS_TOKENS = 576
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEBIAS_STATS_DIR = _PROJECT_ROOT / "myPlayground/emptyTest/output/description/csv_output"


def _load_mean_scores(layer_idx: int, csv_dir: str) -> torch.Tensor:
    csv_path = os.path.join(
        csv_dir,
        os.environ.get(
            "ATTN_DEBIAS_CSV_TEMPLATE",
            "relation_vis_text_stats_relation_vis_text_empty_with_description_Simplex_64_layer{layer_idx}.csv",
        ).format(layer_idx=layer_idx),
    )
    mean_scores = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "mean" not in reader.fieldnames:
            raise ValueError(f"Missing 'mean' column in attention debias CSV: {csv_path}")
        for row_idx, row in enumerate(reader, 2):
            value = float(row["mean"])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid mean at {csv_path}:{row_idx}: {row['mean']!r}")
            mean_scores.append(value)
    if len(mean_scores) != _EXPECTED_DEBIAS_TOKENS:
        raise ValueError(f"Expected {_EXPECTED_DEBIAS_TOKENS} mean values in {csv_path}, got {len(mean_scores)}")
    return torch.tensor(mean_scores, dtype=torch.float32)


def _get_mean_scores_cached(layer_idx: int, csv_dir: str, device: torch.device) -> torch.Tensor:
    global _MEAN_SCORE_CACHE
    csv_dir = str(Path(csv_dir).resolve())
    cache_key = (layer_idx, csv_dir)
    if cache_key not in _MEAN_SCORE_CACHE:
        _MEAN_SCORE_CACHE[cache_key] = _load_mean_scores(layer_idx, csv_dir)
    return _MEAN_SCORE_CACHE[cache_key].to(device)


def attn_postprocess_topk(
    self_attn_weights: torch.Tensor,
    v_token_start: int,
    v_token_num: int,
    text_token_start: int,
    t_token_idx: torch.Tensor,
    layer_idx: int,
    retained_tokens: int,
    position_ids: torch.Tensor = None,
):
    """Calibrate mean text-to-visual attention at original patch positions."""
    self_attn_weights = self_attn_weights.mean(1)

    # 文本段索引行 (原有行为：取第二维索引并加上text_token_start)
    t_row = t_token_idx[1] + text_token_start
    # 取视觉段注意力： [B, L2, L1] -> 对L2求均值得到 [B, L1]
    # 注意：这里得到的是 [B, Text_Len, Vis_Len] 或类似的中间态，取决于 t_row 的维度
    relation_vis_text = self_attn_weights[:, t_row, v_token_start : v_token_start + v_token_num]

    relation_vis_text = relation_vis_text.mean(1)
    csv_dir = os.environ.get("ATTN_DEBIAS_STATS_DIR", str(_DEBIAS_STATS_DIR))
    mean_scores = _get_mean_scores_cached(layer_idx, csv_dir, relation_vis_text.device)
    if position_ids is not None:
        position_indices = position_ids[0, v_token_start : v_token_start + v_token_num].long() - v_token_start
        mean_scores = mean_scores[position_indices]
    else:
        mean_scores = mean_scores[: relation_vis_text.size(1)]
    relation_vis_text = relation_vis_text / (mean_scores.unsqueeze(0) + EPSILON)

    return relation_vis_text


# ========================================================================================
# PRUNING UTILITY FUNCTIONS (from modelling_sparse_llama.py)
# ========================================================================================


def DivPrune_Weighted(
    visual_feature_vectors: torch.Tensor,
    importance_scores: torch.Tensor,
    image_feature_length: int,
    threshold_ratio: float = None,
    preserve_token_num: int = None,
) -> torch.Tensor:
    """
    Weighted Farthest Point Sampling (Weighted FPS) for visual token pruning.
    Uses mean-scaled calibrated importance and a running minimum distance.

    Args:
        visual_feature_vectors: [N, D] visual feature matrix
        importance_scores: [N] importance scores from attention
        image_feature_length: total number of visual tokens (kept for compatibility)
        threshold_ratio: DEPRECATED - ratio of tokens to keep (use preserve_token_num instead)
        preserve_token_num: Fixed number of tokens to preserve (overrides threshold_ratio)

    Returns:
        selected_indices: [K] indices of selected tokens
    """
    N = visual_feature_vectors.shape[0]

    # Calculate K based on new fixed token count approach
    if preserve_token_num is not None:
        # New approach: Use fixed token count
        K = preserve_token_num
    elif threshold_ratio is not None:
        # Fallback to old ratio-based approach for backward compatibility
        K = int(round(threshold_ratio * image_feature_length))
    else:
        raise ValueError("Either preserve_token_num or threshold_ratio must be provided")

    # Ensure K is in valid range [1, N]
    K = max(1, min(K, N))

    device = visual_feature_vectors.device

    # Compute distance matrix (1 - cosine similarity) for geometry
    # Use .contiguous() to optimize matrix multiplication performance
    vis_tokens_contig = visual_feature_vectors.contiguous()
    norm_matrix = vis_tokens_contig / (vis_tokens_contig.norm(dim=1, keepdim=True) + 1e-6)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    distance_matrix = 1.0 - cosine_similarity  # [N, N]

    # -------------------------------------------------------------------------
    # Mean-based significance scaling
    # -------------------------------------------------------------------------
    # 1. Convert to float for calculation
    scores = importance_scores.float()

    # 2. Calculate Mean
    mean_score = torch.mean(scores)

    # 3. Normalize by dividing by mean
    # Adding 1e-6 to denominator to prevent division by zero
    importance_norm = scores / (2 * mean_score + 1e-6)

    # Initialize selected indices
    selected_indices = torch.zeros(K, dtype=torch.long, device=device)

    # Step 1: Select the token with highest importance score as anchor
    # Note: We use original scores for the anchor to ensure the absolute best token is kept.
    selected_indices[0] = torch.argmax(importance_scores)
    min_dist_to_selected = distance_matrix[:, selected_indices[0]].clone()
    for i in range(1, K):
        metric = 0.5 * importance_norm + 0.5 * min_dist_to_selected
        metric[selected_indices[:i]] = -float('inf')
        selected_indices[i] = torch.argmax(metric)
        min_dist_to_selected = torch.minimum(min_dist_to_selected, distance_matrix[:, selected_indices[i]])
    return selected_indices


# ========================================================================================
# ORIGINAL UTILITY FUNCTIONS
# ========================================================================================


def batch_index_select(x, idx):
    if len(x.size()) == 4:
        B, H, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, H, C)[idx.reshape(-1)].reshape(B, H, N_new, C)
        return out
    elif len(x.size()) == 3:
        # in this condition
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


def softmax_with_policy(attn, policy, eps=1e-6):  # attn : [2, 687, 32, 32] policy : [2, 687, 1]
    B, N, _ = policy.size()
    B, H, T, N = attn.size()
    if T == 1:
        # attn_policy = policy.reshape(B, 1, 1, N)
        # max_att = torch.max(attn, dim=-1, keepdim=True)[0]  # [2, 32, 687, 1]
        # attn = attn - max_att
        # attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)  # [2, 32, 687, 687]
        # attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)      # [2, 32, 687, 687]
        policy_bias = torch.zeros(B, 1, N, 1, dtype=policy.dtype).to(device=policy.device)
        policy_bias.masked_fill_(policy.logical_not(), float("-inf"))
        policy_bias = policy_bias.permute(0, 1, 3, 2).to(policy.dtype)
        attn += policy_bias.to(device=attn.device)
        attn = torch.softmax(attn, dim=-1)
        return attn
    else:
        # attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)    [2, 1, 1, 687]
        # eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N) # [1, 1, 687, 687]
        # attn_policy = attn_policy + (1.0 - attn_policy) * eye   # [2, 1, 687, 687]
        # max_att = torch.max(attn, dim=-1, keepdim=True)[0]  # [2, 32, 687, 1]
        # attn = attn - max_att   #
        # # attn = attn.exp_() * attn_policy
        # # return attn / attn.sum(dim=-1, keepdim=True)

        # # for stable training
        # attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)  # [2, 32, 687, 687]
        # attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)      # [2, 32, 687, 687]
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)    [2, 1, 1, 687]
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)  # [1, 1, 687, 687]
        attn_policy = attn_policy + (1.0 - attn_policy) * eye  # [2, 1, 687, 687]
        policy_bias = torch.zeros(B, 1, N, N, dtype=attn_policy.dtype).to(device=attn_policy.device)
        policy_bias.masked_fill_(attn_policy.logical_not(), float("-inf"))
        policy_bias.to(attn_policy.dtype)
        attn += policy_bias
        attn = torch.softmax(attn, dim=-1)
        return attn


def scaled_dot_product_attention_with_policy(
    query, key, value, policy, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    if is_causal:
        assert attn_mask is None
        # attn_bias = torch.zeros(L, S, dtype=query.dtype)
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        attn_bias = torch.zeros(attn_mask.shape, dtype=query.dtype).to(device=query.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(device=query.device)
    attn_weight = softmax_with_policy(attn_weight, policy)
    attn_logits = attn_weight.clone().detach()

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_logits


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(query.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_logits = attn_weight

    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value, attn_logits


def sparse_scaled_dot_product_attention(
    query, key, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1)
    attn_weight += attn_bias.to(query.device)
    attn_logits = attn_weight

    return attn_logits
