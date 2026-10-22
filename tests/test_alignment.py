"""CPU tests for BOND selection, attention statistics, and spatial alignment.

Run with ``python -m unittest discover -s tests -v`` in either model environment.
The tests load individual functions without loading Transformers or a checkpoint.
"""

from __future__ import annotations

import ast
import copy
import csv
import importlib.util
import json
import math
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
QWEN = "Qwen2_5-VL/Qwen2_5VL/modeling_qwen2_5_vl_self.py"
QWEN_REF = "Qwen2_5-VL/Qwen2_5VL/modeling_qwen2_5_vl_self_full_test.py"


def functions(path, names, class_name=None):
    """Load real function bodies without importing their model dependencies."""
    tree = ast.parse((ROOT / path).read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [copy.deepcopy(node) for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise AssertionError(f"Missing functions in {path}: {names}")
    for node in selected:
        node.decorator_list = []
    namespace = {
        "torch": torch,
        "F": F,
        "math": math,
        "os": os,
        "csv": csv,
        "_VisualRegionInfo": types.SimpleNamespace,
        "EPSILON": 1e-6,
        "BASE_GRID_SIDE": 23,
        "_MEAN_SCORE_CACHE": {},
    }
    source = ast.Module(body=ast.parse("from __future__ import annotations").body + selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(source), str(ROOT / path), "exec"), namespace)
    return namespace


Q = functions(QWEN, ["repeat_kv"])
Q.update(functions(QWEN, ["_compute_visual_scores"], "Qwen2_5_VLAttention"))
Q.update(
    functions(
        QWEN,
        [
            "DivPrune_Weighted",
            "_detect_visual_region",
            "get_visual_importance_score",
            "_get_interpolated_mean_scores",
            "forward",
        ],
        "DART",
    )
)
# Each extracted function keeps its own globals dictionary.
Q["_compute_visual_scores"].__globals__["repeat_kv"] = Q["repeat_kv"]

spec = importlib.util.spec_from_file_location("bond_llava_utils", ROOT / "llava/model/language_model/utils.py")
LLAVA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LLAVA)
LLAVA_REF = functions("llava/model_testFull/language_model/score.py", ["reference_scores"])["reference_scores"]


def reference_probabilities():
    """Execute the offline recorder's actual probability calculation."""
    tree = ast.parse((ROOT / QWEN_REF).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen2_5_VLAttention")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    branch = next(
        n
        for n in forward.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "return_logits"
    )
    wrapper = ast.parse("def calculate(self, query_states, key_states, attention_mask, q_len): pass").body[0]
    wrapper.body = copy.deepcopy(branch.body) + [ast.Return(value=ast.Name(id="attn_logits", ctx=ast.Load()))]
    namespace = {"torch": torch, "repeat_kv": Q["repeat_kv"]}
    module = ast.Module(body=[wrapper], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(ROOT / QWEN_REF), "exec"), namespace)
    return namespace["calculate"]


class AlignmentTests(unittest.TestCase):
    def test_llava_reference_attention_keeps_original_positions(self):
        namespace = functions(
            "llava/model_testFull/language_model/modelling_sparse_llama.py",
            ["forward"],
            "LlamaDynamicvitFlashAttention2",
        )
        namespace.update(
            {
                "apply_rotary_pos_emb": lambda query, key, cos, sin, positions: (query, key),
                "repeat_kv": Q["repeat_kv"],
            }
        )
        projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(4))
        rotary_lengths = []

        def rotary(value, seq_len):
            rotary_lengths.append(seq_len)
            return None, None

        module = types.SimpleNamespace(
            q_proj=projection,
            k_proj=projection,
            v_proj=projection,
            o_proj=projection,
            num_heads=2,
            num_key_value_heads=2,
            head_dim=2,
            hidden_size=4,
            num_key_value_groups=1,
            rotary_emb=rotary,
            is_causal=True,
            training=False,
            attention_dropout=0,
            config=types.SimpleNamespace(),
            _flash_attention_forward=lambda query, *args, **kwargs: torch.zeros_like(query),
        )
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 24
        positions = torch.tensor([[0, 1, 2, 7, 8, 9]])
        rows = (torch.zeros(2, dtype=torch.long), torch.arange(2))
        output = namespace["forward"](
            module,
            hidden,
            position_ids=positions,
            return_logits=True,
            v_token_start=2,
            v_token_num=2,
            t_token_idx=rows,
        )
        self.assertEqual(rotary_lengths, [10])
        query = hidden.reshape(1, 6, 2, 2).transpose(1, 2)
        expected_logits = query @ query.transpose(2, 3) / math.sqrt(2)
        expected_logits.masked_fill_(~torch.ones(6, 6, dtype=torch.bool).tril(), float("-inf"))
        torch.testing.assert_close(output[-1], expected_logits.softmax(-1))

    def test_reference_recorders_keep_all_tokens_and_record_every_layer(self):
        cases = [
            ("llava/model_testFull/language_model/modelling_sparse_llama.py", "LlamaDynamicvitModel", 32),
            (QWEN_REF, "DART", 28),
        ]
        for path, cls, layer_count in cases:
            with self.subTest(backbone=cls), tempfile.TemporaryDirectory() as directory:
                namespace = functions(path, ["forward"], cls)
                namespace["reference_scores"] = LLAVA_REF
                namespace["json"] = json
                config = types.SimpleNamespace(
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=False,
                    use_return_dict=False,
                    image_token_id=99,
                )
                ids, positions, grid = self.spatial_input(2)
                hidden = torch.ones(1, 10, 4)
                raw = (
                    torch.zeros(1, 1, 10, 10)
                    .masked_fill(~torch.ones(10, 10, dtype=torch.bool).tril(), float("-inf"))
                    .softmax(-1)
                )
                lengths = []

                class Decoder:
                    attention_type = "full_attention"

                    def __call__(self, hidden_states, **kwargs):
                        lengths.append(hidden_states.shape[1])
                        return hidden_states, raw

                model = types.SimpleNamespace(
                    config=config,
                    gradient_checkpointing=False,
                    training=False,
                    _use_flash_attention_2=True,
                    _use_sdpa=False,
                    spatial_merge_size=2,
                    has_sliding_layers=False,
                    layers=[Decoder() for _ in range(layer_count)],
                    rotary_emb=lambda h, p: (None, None),
                    norm=lambda x: x,
                )
                kwargs = dict(inputs_embeds=hidden, position_ids=positions, use_cache=False, return_dict=False)
                if cls == "DART":
                    model.get_visual_importance_score = types.MethodType(
                        functions(QWEN_REF, ["get_visual_importance_score"], "DART")["get_visual_importance_score"],
                        model,
                    )
                    kwargs.update(input_ids=ids, image_grid_thw=grid, attention_mask={"full_attention": None})
                else:
                    kwargs.update(image_shape=6, pre_prompt_length_list=[2])
                output_path = Path(directory) / "scores.jsonl"
                with patch.dict(os.environ, {"REFERENCE_SCORE_FILE": str(output_path)}):
                    output = namespace["forward"](model, **kwargs)
                records = [json.loads(line) for line in output_path.read_text().splitlines()]
                self.assertEqual([row["layer_idx"] for row in records], list(range(layer_count)))
                for row in records:
                    score = torch.tensor(row["relation_vis_text"])
                    self.assertEqual(score.numel(), 6)
                    torch.testing.assert_close(score.reshape(-1), raw[:, :, 8:, 2:8].mean(dim=(0, 1, 2)))
                self.assertEqual(lengths, [10] * layer_count)
                self.assertEqual(output[0].shape[1], 10)

    def test_qwen_online_matches_reference_with_and_without_masks(self):
        generator = torch.Generator().manual_seed(17)
        query = torch.randn(1, 4, 8, 3, generator=generator)
        key = torch.randn(1, 2, 8, 3, generator=generator)
        module = types.SimpleNamespace(num_key_value_groups=2, scaling=1 / math.sqrt(3))
        causal = torch.ones(8, 8, dtype=torch.bool).tril()[None, None]
        additive = torch.zeros(1, 1, 8, 8).masked_fill(~causal, float("-inf"))
        offline = reference_probabilities()
        results = []
        for mask in [None, causal, additive]:
            with self.subTest(mask_type=str(None if mask is None else mask.dtype)):
                full = offline(module, query, key, mask, 8)
                expected = full[:, :, 5:, 2:5].mean(dim=(0, 1, 2))
                actual = Q["_compute_visual_scores"](module, query, key, mask, 2, 5, 1, 8)
                torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
                results.append(actual)
        torch.testing.assert_close(results[0], results[1])
        torch.testing.assert_close(results[0], results[2])

    def test_qwen_future_keys_do_not_change_earlier_scores(self):
        query = torch.ones(1, 1, 6, 2)
        key = torch.zeros(1, 1, 6, 2)
        key[:, :, 5] = 100
        module = types.SimpleNamespace(num_key_value_groups=1, scaling=1)
        score = Q["_compute_visual_scores"](module, query, key, None, 1, 3, 1, 6)
        # Queries 3 and 4 attend uniformly to 4 and 5 visible keys; query 5
        # attends to its high-logit final key. Future keys cannot affect 3/4.
        expected = torch.full((2,), (1 / 4 + 1 / 5) / 3)
        torch.testing.assert_close(score, expected)

    def test_llava_reference_and_online_use_the_same_raw_statistic(self):
        raw = torch.arange(2 * 7 * 7, dtype=torch.float32).reshape(1, 2, 7, 7)
        raw = raw.masked_fill(~torch.ones(7, 7, dtype=torch.bool).tril(), float("-inf"))
        probabilities = raw.softmax(-1)
        rows = (torch.zeros(3, dtype=torch.long), torch.arange(3))
        reference = LLAVA_REF(probabilities, 1, 3, 4, rows)
        with patch.object(LLAVA, "_get_mean_scores_cached", return_value=torch.ones(576)):
            online = LLAVA.attn_postprocess_topk(probabilities, 1, 3, 4, rows, 0, 3)
        torch.testing.assert_close(online, reference / (1 + LLAVA.EPSILON))

    def test_llava_reference_lookup_preserves_original_patch_positions(self):
        probabilities = torch.full((1, 2, 6, 6), 1 / 6)
        rows = (torch.zeros(2, dtype=torch.long), torch.arange(2))
        # After pruning, patch indices 0 and 5 occupy the two visual slots.
        positions = torch.tensor([[0, 1, 2, 7, 8, 9]])
        reference = torch.arange(1, 577, dtype=torch.float32)
        with patch.object(LLAVA, "_get_mean_scores_cached", return_value=reference):
            actual = LLAVA.attn_postprocess_topk(probabilities, 2, 2, 4, rows, 0, 2, positions)
        expected = torch.tensor([[1 / 6, 1 / 36]])
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-5)

    def test_llava_default_csv_filename_is_simplex(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            path = (
                Path(directory)
                / "relation_vis_text_stats_relation_vis_text_empty_with_description_Simplex_64_layer0.csv"
            )
            path.write_text("mean\n" + "1\n" * 576)
            self.assertEqual(LLAVA._load_mean_scores(0, directory).tolist(), [1] * 576)

    def test_running_minimum_preserves_greedy_selection_and_tie_order(self):
        generator = torch.Generator().manual_seed(29)
        for n in [1, 3, 11]:
            for tied in [False, True]:
                features = torch.ones(n, 5) if tied else torch.randn(n, 5, generator=generator)
                scores = torch.ones(n) if tied else torch.rand(n, generator=generator)
                normalized = features / (features.norm(dim=1, keepdim=True) + 1e-6)
                distance = 1 - normalized @ normalized.T
                importance = scores / (2 * scores.mean() + 1e-6)
                for budget in [1, max(1, n // 2), n, n + 2]:
                    k = min(budget, n)
                    selected = [int(scores.argmax())]
                    while len(selected) < k:
                        metric = importance + distance[:, selected].min(dim=1).values
                        metric[selected] = -float("inf")
                        selected.append(int(metric.argmax()))
                    actual_qwen = Q["DivPrune_Weighted"](features, scores, budget).tolist()
                    actual_llava = LLAVA.DivPrune_Weighted(features, scores, n, preserve_token_num=budget).tolist()
                    self.assertEqual(actual_qwen, selected)
                    self.assertEqual(actual_llava, selected)

    @staticmethod
    def spatial_input(offset):
        ids = torch.tensor([[1, 2, 99, 99, 99, 99, 99, 99, 3, 4]])
        positions = torch.zeros(3, 1, 10, dtype=torch.long)
        positions[1, 0, 2:8] = torch.tensor([0, 0, 0, 1, 1, 1]) + offset
        positions[2, 0, 2:8] = torch.tensor([0, 1, 2, 0, 1, 2]) + offset
        return ids, positions, torch.tensor([[1, 4, 6]])

    def test_qwen_grid_and_lookup_are_invariant_to_text_prefix_offset(self):
        heatmap = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(image_token_id=99, video_token_id=100), spatial_merge_size=2
        )
        model._get_interpolated_mean_scores = lambda layer, h, w, device: heatmap.to(device)
        for offset in [0, 7, 35]:
            ids, positions, grid = self.spatial_input(offset)
            region = Q["_detect_visual_region"](model, ids, positions, grid, None)
            self.assertEqual(region.true_spatial_size, (2, 3))
            self.assertEqual(region.spatial_origin, (offset, offset))
            # Keep only patches (1,1) and (1,2), removing the minimum row/column.
            keep = torch.tensor([0, 1, 6, 7, 8, 9])
            aligned = Q["get_visual_importance_score"](
                model,
                torch.ones(2),
                torch.ones(1, 6, 4),
                positions[:, :, keep],
                2,
                4,
                0,
                image_grid_thw=grid,
                true_grid_size=region.true_spatial_size,
                spatial_origin=region.spatial_origin,
            )
            torch.testing.assert_close(aligned, 1 / (heatmap[1, 1:] + 1e-6))

    def test_qwen_bilinear_reference_alignment_on_rectangular_grid(self):
        interpolate = Q["_get_interpolated_mean_scores"]
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "reference_0.csv"
            csv_path.write_text("mean\n" + "\n".join(str(10 + 3 * y + 2 * x) for y in range(23) for x in range(23)))
            with patch.dict(
                interpolate.__globals__,
                {"CSV_TEMPLATE": str(Path(directory) / "reference_{}.csv"), "_MEAN_SCORE_CACHE": {}},
            ):
                actual = interpolate(0, 2, 3, torch.device("cpu"))
            ys = (torch.arange(2) + 0.5) * 23 / 2 - 0.5
            xs = (torch.arange(3) + 0.5) * 23 / 3 - 0.5
            expected = 10 + 3 * ys[:, None] + 2 * xs[None, :]
            torch.testing.assert_close(actual, expected)

    def test_sparse_false_disables_all_stages_including_after_enabled_run(self):
        ids, positions, grid = self.spatial_input(2)
        config = types.SimpleNamespace(
            DART_config={"Sparse": True},
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            use_return_dict=False,
            image_token_id=99,
            video_token_id=100,
        )
        events = []

        class Decoder:
            attention_type = "full_attention"

            def __init__(self, index):
                self.index = index

            def __call__(self, hidden_states, return_logits=False, **kwargs):
                events.append((self.index, hidden_states.shape[1], return_logits))
                score = None
                if return_logits is not False:
                    start, end = return_logits
                    score = torch.arange(end - start, dtype=torch.float32) + 1
                return hidden_states + 0.01, score

        model = types.SimpleNamespace(
            config=config,
            pruning_schedule={},
            prune_layers=set(),
            dart_init_done=False,
            gradient_checkpointing=False,
            training=False,
            has_sliding_layers=False,
            spatial_merge_size=2,
            layers=[Decoder(i) for i in range(28)],
            rotary_emb=lambda h, p: (None, None),
            norm=lambda x: x,
        )
        model._detect_visual_region = types.MethodType(Q["_detect_visual_region"], model)
        model.DivPrune_Weighted = Q["DivPrune_Weighted"]
        model.get_visual_importance_score = lambda **kwargs: kwargs["visual_scores"]
        embedding = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4) + 1
        kwargs = dict(
            input_ids=ids,
            inputs_embeds=embedding,
            position_ids=positions,
            attention_mask={"full_attention": None},
            image_grid_thw=grid,
            use_cache=False,
            return_dict=False,
        )
        with patch.dict(os.environ, {"RETENTION_RATIO": "0.25"}):
            output = Q["forward"](model, **kwargs)
        self.assertEqual(model.prune_layers, {0, 2, 26})
        self.assertEqual([index for index, _, score in events if score is not False], [0, 2])
        self.assertEqual(output[0].shape[1], 4)
        config.DART_config = None
        events.clear()
        output = Q["forward"](model, **kwargs)
        self.assertEqual(model.prune_layers, set())
        self.assertTrue(all(score is False and length == 10 for _, length, score in events))
        self.assertEqual(output[0].shape[1], 10)

    def test_sparse_boolean_model_arguments(self):
        configure = functions("lmms-eval/lmms_eval/models/simple/qwen2_5_vl.py", ["configure_DART"])["configure_DART"]
        model = types.SimpleNamespace(config=types.SimpleNamespace(text_config=types.SimpleNamespace()))
        for value, expected in [(False, False), ("False", False), (True, True), ("True", True)]:
            configure(model, {"Sparse": value})
            self.assertEqual(model.config.text_config.DART_config is not None, expected)


if __name__ == "__main__":
    unittest.main()
